import os
from pathlib import Path
import sys
import time
import json
import shutil
import random

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import tqdm

sys.path.append(str(Path(str(os.getcwd())).resolve()))

from utils.logger import logger
from utils.utils import *
from src.common.param import args, model_args, data_args
from env_uav import AirVLNENV
from src.vlnce_src.closeloop_util_nohelp import EvalBatchState, BatchIterator, setup, CheckPort, initialize_env_eval, is_dist_avail_and_initialized

import numpy as np



import copy
import numpy as np
import torch
from PIL import Image
import json
from src.common.param import model_args, args

from transformers import AutoProcessor,AutoModelForImageTextToText
import ast
import re
from scipy.spatial.transform import Rotation as R 


class DinoMonitor:
    _instance = None
    
    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls):
        if not cls._instance:
            cls._instance = DinoMonitor()
            return cls._instance
        return cls._instance
        
    def __init__(self, device=0):
        self.dino_model = None
        self.init_dino_model(device)
        self.object_desc_dict = dict()
        self.init_object_dict()
        
    def init_object_dict(self):
        with open(args.object_name_json_path, 'r') as f:
            file = json.load(f)
            for item in file:
                self.object_desc_dict[item['object_name']] = item['object_desc']
    
    def init_dino_model(self, device):
        import src.model_wrapper.utils.GroundingDINO as GroundingDINO
        import sys
        from functools import partial
        sys.path.append(GroundingDINO.__path__[0])
        from src.model_wrapper.utils.GroundingDINO.groundingdino.util.inference import load_model, predict
        device = torch.device(device)
        model = load_model(model_args.groundingdino_config, model_args.groundingdino_model_path)
        model.to(device=device)
        model.eval()
        self.dino_model = partial(predict, model=model)
    
    def get_dino_results(self, episode, obj_info):
        images = episode[-1]['rgb_record']
        depths = episode[-1]['depth_record']
        done = False
        camera_num = len(images)
        detected_camera = {}
        for i in range(camera_num):
            detected_camera[i]=None
       
        for i in range(len(images)):
            img = images[i]
            depth = depths[i]
            target_detections = []
            boxes, logits = self.detect(img, obj_info)
            print("obj_info",obj_info)

            depth = cv2.resize(
                depth,
                (1024, 1024),
                interpolation=cv2.INTER_NEAREST
            )
            
            if len(boxes) > 0:
                for j, point in enumerate(boxes):
                    point = list(map(int, point))
                    center_point = (int((point[0] + point[2]) / 2), int((point[1] + point[3]) / 2))
                    depth_data = int(depth[center_point[1], center_point[0]] / 2.55)
                    if depth_data < args.PREDICTSUCCESSDISTANCE:
                        target_detections.append((float(logits[j]), depth_data))

            if len(target_detections) > 0:
                done = True
                break

            print("boxes",boxes)
        
            if len(boxes) == 1:
                detected_camera[i]=boxes

        return done,detected_camera
    
    def detect(self, img, prompt):
        import groundingdino.datasets.transforms as T
        from groundingdino.util import box_ops
        
        img_src = copy.deepcopy(np.array(img))
        img = Image.fromarray(img_src)
        transform = T.Compose(
        [   T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        image_transformed, _ = transform(img, None)
        boxes, logits, phrases = self.dino_model(
            image=image_transformed,
            caption=prompt,
            box_threshold=0.6,
            text_threshold=0.40
        )
        logits = logits.detach().cpu().numpy()
        H, W, _ = img_src.shape
        boxes_xyxy = (box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])).cpu().numpy()
        boxes = []
        for box in boxes_xyxy:
            if (box[2] - box[0]) / W > 0.6 or (box[3] - box[1]) / H > 0.5:
                continue
            boxes.append(box)
        return boxes, logits


def predict_done(dino_moinitor, episodes, object_infos):
    prediction_dones = []
    detected_camera_list = []
    for i in range(len(episodes)):
        prediction_done,detected_camera = dino_moinitor.get_dino_results(episodes[i], object_infos[i])
        prediction_dones.append(prediction_done)
        detected_camera_list.append(detected_camera)
    return prediction_dones,detected_camera_list

import numpy as np
from dataclasses import dataclass

# =========================
# 1. 相机模型
# =========================
@dataclass
class CameraModel:
    width: int = 1024
    height: int = 1024
    fov_deg: float = 90.0

    def __post_init__(self):
        self.cx = self.width / 2
        self.cy = self.height / 2
        fov_rad = np.deg2rad(self.fov_deg)
        self.f = self.width / (2 * np.tan(fov_rad / 2))

    def pixel_to_camera_ray(self, u, v):
        """
        像素坐标 -> 相机坐标系方向向量
        Camera frame:
          x: right
          y: down
          z: forward
        """
        x = (u - self.cx) / self.f
        y = (v - self.cy) / self.f
        z = 1.0
        v_cam = np.array([x, y, z])
        return v_cam / np.linalg.norm(v_cam)


# =========================
# 2. 方向估计器
# =========================
class DirectionEstimator:
    def __init__(self):


        self.R_cam_to_body = np.array([
            [0, 0, 1],  # x_body
            [1, 0, 0],  # y_body
            [0, 1, 0],  # z_body
        ])


    def camera_ray_to_body(self, v_cam):
        return self.R_cam_to_body @ v_cam

    def compute_angles(self, v_body):
        """
        v_body: unit vector in body frame
        Returns:
            yaw_deg: 左负右正
            pitch_deg: 下为正
        """
        vx, vy, vz = v_body
        yaw = np.degrees(np.arctan2(vy, vx))
        pitch = np.degrees(np.arctan2(vz, vx))
        return yaw, pitch

    def classify_direction(self, v_body):
        """
        根据你的规则进行分类
        """
        yaw, _ = self.compute_angles(v_body)
        vz = v_body[2]


        direction = None
        # 前 / 左 / 右
        if -15 <= yaw <= 15:
            direction = "forward"
        elif yaw > 15:
            direction = "front-right"
        else:
            direction = "front-left"

        return direction

def eval( eval_env: AirVLNENV, eval_save_dir, model, device):
    dino_moinitor = DinoMonitor.get_instance()
    with torch.no_grad():
        dataset = BatchIterator(eval_env)
        end_iter = len(dataset)
        pbar = tqdm.tqdm(total=end_iter)
        cam = CameraModel()
        estimator_front = DirectionEstimator()
        while True:
            env_batchs = eval_env.next_minibatch()
            traj_name = [b["seq_name"] for b in env_batchs]
            if env_batchs is None:
                break
            
            # 初始化batch_state
            batch_state = EvalBatchState(batch_size=eval_env.batch_size, env_batchs=env_batchs, env=eval_env)
            
            pbar.update(n=eval_env.batch_size)

            for t in range(int(args.maxWaypoints) + 1):
                logger.info('Step: {} \t Completed: {} / {}'.format(t, int(eval_env.index_data)-int(eval_env.batch_size), end_iter))

                is_terminate = batch_state.check_batch_termination(t)
                if is_terminate:
                    break
                cur_pose =[]
                cur_ori=[]
   
                for i in range(eval_env.batch_size):
                    cur_pose.append(eval_env.sim_states[i].pose[:3])
                    cur_ori.append(eval_env.sim_states[i].pose[-4:])


                # 获取当前的图像信息
                prompts = []
                images_list = []
                for i in range(eval_env.batch_size):
                    data = batch_state.episodes[i][-1]
                    if batch_state.updata_direction[i] is not None:
                        description = data["instruction"]
                        b = description.split(".")
                        sign = -3
                        c = b[0:sign]+[batch_state.updata_direction[i]]+b[sign+1:]  
                        description = ".".join(c)
                    else:
                        description = data["instruction"]
                    
                    front_img = Image.fromarray(data["rgb_record"][0])
                    down_img = Image.fromarray(data["rgb_record"][1])
                    images_list.append([front_img, down_img])
                    
                    # 构建消息
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "image"},
                                {"type": "text", "text": description}
                            ]
                        }
                    ]
                    # 生成prompt
                    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    prompts.append(prompt)
                    
                inputs = processor(
                    text=prompts, 
                    images=images_list,  # 传入列表的列表
                    return_tensors="pt",
                    padding=True  # 启用填充
                )
                    # 转移到GPU
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(device)

     
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)

                # 解码结果
                results = processor.batch_decode(outputs, skip_special_tokens=True)
                print(results)
                traj = []
                for i, result in enumerate(results):
                    match = re.search(r'\[\[.*\]\]', result)  # 匹配最外层的双中括号
                    if match:
                        lst_str = match.group()
                        lst = ast.literal_eval(lst_str)
                        traj.append(lst)
                    else:
                        print("No list found")

                global_taj = []
                for i in range(eval_env.batch_size):
                    local_traj = traj[i]


                    P_t = np.array(cur_pose[i])
                    q_t = np.array(cur_ori[i])   # (x, y, z, w)
                    R_t = R.from_quat(q_t).as_matrix()  # body -> world
        
                    pred_world = []
                    for dp_body in local_traj:  # dp_body: 单个相对位移
                        dp_world = R_t @ dp_body        # body -> world
                        p_future = P_t + dp_world       # 加上当前全局位置
                        pred_world.append(p_future)
                    global_taj.append(pred_world)


                next_waypoints = global_taj

                eval_env.makeActions_qwenvl(next_waypoints)
                outputs = eval_env.get_obs()
                batch_state.update_from_env_output(outputs)
                obj_des = [batch_state.traj_objdes[traj] for traj in traj_name]
                if args.use_target_des:
                    batch_state.predict_dones, _ = predict_done(dino_moinitor, batch_state.episodes, batch_state.object_infos)
                    _, detected_camera_list_direction = predict_done(dino_moinitor, batch_state.episodes, obj_des)
                else:
                    batch_state.predict_dones, detected_camera_list_direction = predict_done(dino_moinitor, batch_state.episodes, batch_state.object_infos)
                for i in range(eval_env.batch_size):
                    detected_camera = detected_camera_list_direction[i]
                    for camera, box in detected_camera.items():
                        if camera == 0:
                            if box is not None:
                                point = list(map(int, box[0]))
                                center_point = (int((point[0] + point[2]) / 2), int((point[1] + point[3]) / 2))
                                u = center_point[0] # x
                                v = center_point[1] # y
                                v_cam = cam.pixel_to_camera_ray(u, v)
                                v_body = estimator_front.camera_ray_to_body(v_cam)
                                direction = estimator_front.classify_direction(v_body)
                                if direction == "forward":
                                    direction_des = "It's straight ahead"
                                elif direction == "front-right":
                                    direction_des = "It's off to the front-right"
                                elif direction == "front-left":
                                    direction_des = "It's off to the front-left"
                                batch_state.updata_direction[i] = direction_des
    
                        # 优先为下
                        if camera == 1:
                            if box is not None:
                                direction = "It's down below you"
                                batch_state.updata_direction[i] = direction
    
                batch_state.update_metric()
        try:
            pbar.close()
        except:
            pass


if __name__ == "__main__":
    
    eval_save_path = args.eval_save_path
    eval_json_path = args.eval_json_path
    dataset_path = args.dataset_path
    port = args.simulator_tool_port

    # --------------------------

    merged_model_path = model_args.model_path

    # --------------------------
    # 加载 processor
    # --------------------------
    processor = AutoProcessor.from_pretrained(
        merged_model_path,
        trust_remote_code=True
    )

    
    model = AutoModelForImageTextToText.from_pretrained(
        merged_model_path,
        device_map="auto",
        dtype=torch.float16,
        trust_remote_code=True
    )

    model.eval()
    device = model.device
    
    if not os.path.exists(eval_save_path):
        os.makedirs(eval_save_path)
    
    setup()

    assert CheckPort(), 'error port'

    eval_env = initialize_env_eval(dataset_path=dataset_path, save_path=eval_save_path, eval_json_path=eval_json_path, port=port)

    if is_dist_avail_and_initialized():
        torch.distributed.destroy_process_group()

    args.DistributedDataParallel = False
    
    eval(
         eval_env=eval_env,
         eval_save_dir=eval_save_path,model=model,device=device)
    
    eval_env.delete_VectorEnvUtil()

