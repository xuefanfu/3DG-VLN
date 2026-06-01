import copy
import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Sequence
import transformers
from scipy.spatial.transform import Rotation as R
import torch
import numpy as np
import math


sys.path.append(str(Path(str(os.getcwd())).resolve()))
sys.path.append(str(Path(__file__).resolve().parents[3]/ 'Model' / 'LLaMA-UAV'))
from llamavid.model.builder import load_pretrained_model
from llamavid.model.vis_traj_arch import VisionTrajectoryGenerator
from peft import PeftModel
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from llamavid.constants import (
    IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
    WAYPOINT_INPUT_TOKEN, WAYPOINT_LABEL_TOKEN, DEFAULT_HISTORY_TOKEN, DEFAULT_WP_TOKEN
)
from llamavid import conversation as conversation_lib
def load_model(args):
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, args.model_base, model_name)
    
    smarter_tokenizer_and_embedding_resize(special_tokens_list=['<wp>', '<his>'], tokenizer=tokenizer, model=model)
    model.get_special_token_id({'<wp>': tokenizer.encode('<wp>')[1], '<his>': tokenizer.encode('<his>')[1],
                                ',': tokenizer.encode(',')[1], ';': tokenizer.encode(';')[1]})
    lora_enable = True
    if lora_enable:
        print(f"Loading LoRA weights from {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
        non_lora_weights = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
        model.load_state_dict(non_lora_weights, strict=False)    
        mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
        model.load_state_dict(mm_projector_weights, strict=False)
    
    return tokenizer, model, image_processor

def load_traj_model(model_args):
    vision_config = generate_vision_tower_config(model_args.vision_tower, model_args.image_processor)
    config = transformers.AutoConfig.from_pretrained(vision_config, trust_remote_code=True)
    traj_model = VisionTrajectoryGenerator(config)
    traj_weights = torch.load(os.path.join(model_args.traj_model_path, 'model_5.pth'), map_location='cpu')
    traj_weights = {k: v.to(torch.bfloat16) for k, v in traj_weights.items()}
    traj_model.load_state_dict(traj_weights, strict=False)
    return traj_model


def generate_vision_tower_config(vision_tower, image_processor):
    default_vision_config={
    "model_type": "clip",
    "hidden_act": "silu",
    "hidden_size": 4096,
    "image_aspect_ratio": "square",
    "image_grid_pinpoints": None,
    "image_processor": "./llamavid/processor/clip-patch14-224",
    "initializer_range": 0.02,
    "intermediate_size": 11008,
    "max_position_embeddings": 4096,
    "max_token": 2048,
    "mm_hidden_size": 1408,
    "mm_projector_type": "mlp2x_gelu",
    "mm_use_im_patch_token": False,
    "mm_use_im_start_end": False,
    "mm_vision_select_feature": "patch",
    "mm_vision_select_layer": -2,
    "mm_vision_tower": "./model_zoo/LAVIS/eva_vit_g.pth",
    "torch_dtype": "float16"
    }
    default_vision_config['image_processor'] = image_processor
    default_vision_config['mm_vision_tower'] = vision_tower
    cf_path = os.path.join(os.path.split(vision_tower)[0], 'config.json')
    with open(cf_path, 'w') as f:
        json.dump(default_vision_config, f, indent=2)
    return cf_path


def prepare_data_to_traj_model(episodes, waypoints, image_processor, rot_to_targets=None):
    image_list = []
    target_list = []
    for i in range(len(episodes)):
        info = episodes[i]
        rot_to_target = None
        if rot_to_targets is not None:
            if rot_to_targets[i] is not None:
                rot_to_target = rot_to_targets[i]
        target = waypoints[i][0:3]
        rot_0 = info[0]['sensors']['imu']["rotation"]
        rot = info[-1]['sensors']['imu']["rotation"]
        if rot_to_target is not None:
            target = np.array(rot).T @ np.array(rot_0) @ np.array(rot_to_target) @ np.array(target)
        else:
            target = np.array(rot).T @ np.array(rot_0) @ np.array(target)
        image_list.append(info[-1]['rgb'][0])
        target_list.append(target)
    images = np.stack(image_list, axis=0)
    image = image_processor.preprocess(images, return_tensors='pt')['pixel_values']
    target = torch.tensor(np.array(target_list))
    
    return {'img': image, 'target': target}        

def transform_to_world(waypoints, episodes):
    waypoints_world = []
    for i in range(len(waypoints)):
        waypoint = waypoints[i]
        ep = episodes[i]
        pos = ep[-1]["sensors"]["state"]["position"]
        rot = ep[-1]["sensors"]["imu"]["rotation"]  
        waypoint_world = np.array(rot) @ np.array(waypoint).T + np.asarray(pos).reshape(3,1)
        waypoint_world = waypoint_world.T
        waypoints_world.append(waypoint_world)

    return waypoints_world

def smarter_tokenizer_and_embedding_resize(
    special_tokens_list: List,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_tokens(special_tokens_list, special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def to_eularian_angles(q):
    x,y,z,w = q
    ysqr = y * y
    t0 = +2.0 * (w*x + y*z)
    t1 = +1.0 - 2.0*(x*x + ysqr)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w*y - z*x)
    if (t2 > 1.0):
        t2 = 1
    if (t2 < -1.0):
        t2 = -1.0
    pitch = math.asin(t2)
    t3 = +2.0 * (w*z + x*y)
    t4 = +1.0 - 2.0 * (ysqr + z*z)
    yaw = math.atan2(t3, t4)
    return (pitch, roll, yaw)

def euler_to_rotation_matrix(e):
    rotation = R.from_euler('xyz', e, degrees=False)
    return rotation.as_matrix()

def project_this_state2target_state_axis(this_state, target_state):
    start_pos = target_state['position']
    start_eular = to_eularian_angles(target_state['orientation'])  # (pitch, roll, yaw)
    this_pos = this_state['position']
    this_eular = to_eularian_angles(this_state['orientation'])
    delta_pos = np.asarray(this_pos) - np.asarray(start_pos)
    delta_eular = np.asarray(this_eular) - np.asarray(start_eular)
    rot = euler_to_rotation_matrix(start_eular) 
    delta_pos = rot.T @ delta_pos
    return {'position': delta_pos.tolist(), 'orientation': delta_eular.tolist()}

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    video_folder: Optional[str] = field(default=None)
    video_fps: Optional[int] = field(default=1)
    video_token: Optional[int] = field(default=2)
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    input_prompt: Optional[str] = field(default=None)
    refine_prompt: Optional[bool] = field(default=False)
    mm_use_im_start_end: bool = field(default=False)

@dataclass
class CommonArguments:
    model_path: Optional[str] = field(default="facebook/opt-350m")
    model_base: Optional[str] = field(default=None)

def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments,
    stage = None,
    delta = None,
    cur_pos = None
) -> Dict:
    """
        process image token's representation
    """
    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['prompt'] = copy.deepcopy(sentence['value'])
                sentence['value'] = '\n\nStage:' + stage + '\n\nPrevious displacement:' + delta  + '\n\nCurrent position:' + cur_pos + '\n\nCurrent image:' + DEFAULT_IMAGE_TOKEN + '\n\nInstruction:' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources

def rotation_matrix_from_vector(x, y):
    v_x = np.array([x, y, 0])
    v_x = v_x / np.linalg.norm(v_x)
    v_y = np.array([-v_x[1], v_x[0], 0])
    v_y = v_y / np.linalg.norm(v_y)
    v_z = np.array([0, 0, 1])
    rotation_matrix = np.column_stack((v_x, v_y, v_z))
    return rotation_matrix

def transform_point(point, rotation_matrix):
    return np.dot(point, rotation_matrix)


def prepare_data_to_inputs(episodes, tokenizer, image_processor, data_args, target_point, assist_notice = None):

    ori_sources = None
    input_prompt = data_args.input_prompt
    refine_prompt = data_args.refine_prompt
    sources = episodes
    ori_sources = copy.deepcopy(sources)
    processor = image_processor
    images = []
    for src in sources[::-1]:
        if 'rgb' in src:
            images.extend(src['rgb'])
            break
    images = np.stack(images, axis=0)
    image = processor.preprocess(images, return_tensors='pt')['pixel_values']
    
    conversation_for_human = '<image>\n' + sources[-1]['instruction']
    conversation = [
    {
        "from": "human",
        "value": conversation_for_human},
    {
        "from": "gpt",
        "value": ""
    }]
    
    if assist_notice is not None:
        stage = assist_notice
    else:
        # stage = 'cruise' if len(sources) > 20 else 'take off'
        stage = 'cruise'
    rot = np.array(ori_sources[0]['sensors']['imu']["rotation"])
    pos = np.array(ori_sources[0]['sensors']['state']['position'])
    deltas = []
    for source in ori_sources:
        if 'rgb' not in source.keys():
            continue
        deltas.append((np.array(source['sensors']['state']['position']) - pos))
    history_waypoint = np.array([(rot.T @ delta) for delta in deltas])
    rotation_to_target = None
    
    target_point = np.array(rot.T @ (target_point - pos))
    x, y = target_point[0], target_point[1]
    rotation_to_target = rotation_matrix_from_vector(x, y)
    history_waypoint = transform_point(history_waypoint, rotation_to_target)

    if len(history_waypoint) >= 2:
        delta = history_waypoint[-1] - history_waypoint[-2]
    else:
        delta = np.array([0, 0, -4.5])
    delta = delta / (np.linalg.norm(delta) + 1e-8)
    delta = ','.join([str(round(x, 1)) for x in delta])
    cur_pos = history_waypoint[-1]
    cur_pos = ','.join([str(round(x, 1)) for x in cur_pos])
    # print('stage:', stage,'delta:', delta, 'cur_pos:', cur_pos)
    sources = preprocess_multimodal(copy.deepcopy([conversation]), data_args, stage=stage, delta=delta, cur_pos=cur_pos)
    data_dict = preprocess(
        sources,
        tokenizer,
        has_image=True,
        prompt=input_prompt,
        refine_prompt=refine_prompt)
    if 'prompt' in data_dict:
        prompt = data_dict['prompt']
    else:
        prompt = None
        
    data_dict = dict(input_ids=data_dict["input_ids"][0],
                        labels=data_dict["labels"][0])

    data_dict['image'] = image
    data_dict['history_waypoint'] = torch.tensor(history_waypoint).view(-1)
    ori_0 = ori_sources[0]['sensors']['state']
    ori = ori_sources[-1]['sensors']['state']
    target_relative_orientation = project_this_state2target_state_axis(ori, ori_0)['orientation']
    data_dict['orientation'] =  torch.tensor(target_relative_orientation).view(-1)
    
    if prompt is not None:
        data_dict['prompt'] = prompt
        
    return data_dict, rotation_to_target


def inputs_to_batch(tokenizer, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :tokenizer.model_max_length]
        labels = labels[:, :tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images) and len(images) > 1 and images[0].shape[-1] < 100:
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        if 'prompt' in instances[0]:
            batch['prompts'] = [instance['prompt'] for instance in instances]
        
        if 'history_waypoint' in instances[0]:
            batch['historys'] = [instance['history_waypoint'] for instance in instances]
        
        if 'orientation' in instances[0]:
            batch['orientations'] = torch.stack([instance['orientation'] for instance in instances])

        return batch

def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    prompt: str = None,
    refine_prompt: bool = False,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.version.startswith("imgsp_uav"):
        return preprocess_imgsp_uav(sources, tokenizer, has_image=has_image, refine_prompt=refine_prompt)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)

def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source: 
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def _mask_targets(target, tokenized_lens, speakers):
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len

def preprocess_imgsp_uav(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    img_token: str = '<image>',
    refine_prompt: bool = False,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    guided_prompt = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        img_in_text = False
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            
            # add guided prompt
            if role==conv.roles[0]:
                guided_sent = sentence["prompt"].replace(DEFAULT_IMAGE_TOKEN, '').replace('\n', '')
                if refine_prompt:
                    # only keep the useful part of the prompt
                    object_description = guided_sent.split('degrees from you.')[-1].replace('Please control the drone and find the target.', '').strip()
                    guided_sent = 'Please pay attention to the obstacles in images and approach the object described below: ' + object_description

                guided_prompt.append(guided_sent)
            # check if image token in text
            if img_token in sentence["value"]:
                img_in_text = True
            # add image token to all sentence if multimoal input
            if role==conv.roles[0] and img_in_text and img_token not in sentence["value"]:
                # randomly add image token to the beginning or end of the sentence
                img_conv = img_token + '\n' + sentence["value"]
                
                conv.append_message(role, img_conv)
            else:
                conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    # add wp embedding, input_ids[-1] is </s>, 
    input_ids_pad_wp = torch.zeros(input_ids.shape[0], input_ids.shape[1] + 1, dtype=torch.long)
    input_ids_pad_wp[:, :-2] = input_ids[:, :-1]
    input_ids_pad_wp[:, -2] = WAYPOINT_INPUT_TOKEN
    input_ids_pad_wp[:, -1] = input_ids[:, -1]
    
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX

    targets_pad_wp = torch.zeros(targets.shape[0], targets.shape[1] + 1, dtype=torch.long)
    targets_pad_wp[:, :-2] = targets[:, :-1]
    targets_pad_wp[:, -2] = WAYPOINT_LABEL_TOKEN
    targets_pad_wp[:, -1] = targets[:, -1]

    return dict(
        input_ids=input_ids_pad_wp,
        labels=targets_pad_wp,
        prompt=guided_prompt,
    )