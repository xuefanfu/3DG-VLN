from collections import deque
import multiprocessing
import msgpackrpc
import time
import airsim
import threading
import random
import copy
import numpy as np
import cv2
import os,sys
import math
import tqdm

cur_path=os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, cur_path+"/..")

from utils.logger import logger


class BaseSensor:
    def __init__(self) -> None:
        pass

    def retrieve(self):
        raise NotImplementedError()

class State(BaseSensor):
    def __init__(self, client, drone_name=''):
        self.data = {'position': None, 'linear_velocity': None, 'linear_acceleration':None,
                     'orientation':None, 'angular_velocity':None, 'angular_acceleration':None}
        self.client: airsim.MultirotorClient = client
        self.drone_name = drone_name

    def retrieve(self):
        data = self.client.getMultirotorState(vehicle_name=self.drone_name)
        collision = {}
        collision_info = self.client.simGetCollisionInfo(vehicle_name=self.drone_name)
        collision['has_collided'] = collision_info.has_collided
        collision['object_name'] = data.collision.object_name
        gps_location = [data.gps_location.latitude,data.gps_location.longitude,data.gps_location.altitude]
        timestamp = data.timestamp
        position = list(data.kinematics_estimated.position)
        linear_velocity = list(data.kinematics_estimated.linear_velocity)
        linear_acceleration = list(data.kinematics_estimated.linear_acceleration)
        orientation = list(data.kinematics_estimated.orientation)
        angular_velocity = list(data.kinematics_estimated.angular_velocity)
        angular_acceleration = list(data.kinematics_estimated.angular_acceleration)

        self.data.update({'collision': collision, 
                          'gps_location': gps_location,
                          'timestamp': timestamp, 
                          'position': position,
                          'linear_velocity': linear_velocity,
                          'linear_acceleration': linear_acceleration,
                          'orientation': orientation,
                          'angular_velocity': angular_velocity,
                          'angular_acceleration': angular_acceleration
                          })
        return self.data
        
        
class Imu(BaseSensor):
    def __init__(self, client, drone_name='', imu_name=''):
        self.data = {}
        self.client: airsim.MultirotorClient = client
        self.drone_name = drone_name
        self.imu_name = imu_name

    def retrieve(self):
        data = self.client.getImuData(imu_name=self.imu_name,vehicle_name=self.drone_name)
        time_stamp = data.time_stamp
        orientation = data.orientation
        angular_velocity = list(data.angular_velocity)
        linear_acceleration = list(data.linear_acceleration)
        q0, q1, q2, q3 = orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val
        rotation_matrix = np.array(([1-2*(q2*q2+q3*q3),2*(q1*q2-q3*q0),2*(q1*q3+q2*q0)],
                                      [2*(q1*q2+q3*q0),1-2*(q1*q1+q3*q3),2*(q2*q3-q1*q0)],
                                      [2*(q1*q3-q2*q0),2*(q2*q3+q1*q0),1-2*(q1*q1+q2*q2)])).tolist()
        self.data.update({'time_stamp': time_stamp, 'rotation': rotation_matrix, 'orientation': list(data.orientation),
                          'linear_acceleration': linear_acceleration, 'angular_velocity': angular_velocity})
        return self.data

class MyThread(threading.Thread):
    def __init__(self, func, args):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args
        self.flag_ok = False

    def run(self):
        self.result = self.func(*self.args)
        self.flag_ok = True

    def get_result(self):
        threading.Thread.join(self)
        try:
            return self.result
        except:
            return None


class AirVLNSimulatorClientTool:
    def __init__(self, machines_info) -> None:
        self.machines_info = copy.deepcopy(machines_info)
        self.socket_clients = []
        self.airsim_clients = [[None for _ in list(item['open_scenes'])] for item in machines_info ]
        self.airsim_ports = []
        self.airsim_ip = '127.0.0.1'
        self._init_check()
        self.objects_name_cnt = [[0 for _ in list(item['open_scenes'])] for item in machines_info ]

    def _init_check(self) -> None:
        ips = [item['MACHINE_IP'] for item in self.machines_info]
        assert len(ips) == len(set(ips)), 'MACHINE_IP repeat'

    def _confirmSocketConnection(self, socket_client: msgpackrpc.Client) -> bool:
        try:
            socket_client.call('ping')
            print("Connected\t{}:{}".format(socket_client.address._host, socket_client.address._port))
            return True
        except:
            try:
                print("Ping returned false\t{}:{}".format(socket_client.address._host, socket_client.address._port))
            except:
                print('Ping returned false')
            return False

    def _confirmConnection(self) -> bool:
        for index_1, _ in enumerate(self.airsim_clients):
            for index_2, _ in enumerate(self.airsim_clients[index_1]):
                if self.airsim_clients[index_1][index_2] is not None:
                    confirmed = False
                    count = 0
                    print("confirmed111",confirmed)
                    print("count1111",count)
                    while not confirmed and count < 30:
                        try:
                            self.airsim_clients[index_1][index_2].confirmConnection()
                            print("confirmed222")
                            confirmed = True
                        except Exception as e:
                            time.sleep(1)
                            print('failed', e)
                            count += 1
                            pass
        
        return confirmed

    def _closeSocketConnection(self) -> None:
        socket_clients = self.socket_clients

        for socket_client in socket_clients:
            try:
                socket_client.close()
            except Exception as e:
                pass

        self.socket_clients = []
        return

    def _closeConnection(self) -> None:
        for index_1, _ in enumerate(self.airsim_clients):
            for index_2, _ in enumerate(self.airsim_clients[index_1]):
                if self.airsim_clients[index_1][index_2] is not None:
                    try:
                        self.airsim_clients[index_1][index_2].close()
                    except Exception as e:
                        pass

        self.airsim_clients = [[None for _ in list(item['open_scenes'])] for item in self.machines_info]
        return

    def run_call(self, airsim_timeout: int=300) -> None:
        socket_clients = []
        for index, item in enumerate(self.machines_info):
            socket_clients.append(
                msgpackrpc.Client(msgpackrpc.Address(item['MACHINE_IP'], item['SOCKET_PORT']), timeout=300)
            )

        for socket_client in socket_clients:
            if not self._confirmSocketConnection(socket_client):
                logger.error('cannot establish socket')
                raise Exception('cannot establish socket')

        self.socket_clients = socket_clients


        before = time.time()
        self._closeConnection()

        def _run_command(index, socket_client: msgpackrpc.Client):
            logger.info(f'开始打开场景，机器{index}: {socket_client.address._host}:{socket_client.address._port}')
            logger.info(f'gpus: {self.machines_info[index]}')
            result = socket_client.call('reopen_scenes', socket_client.address._host, list(zip(self.machines_info[index]['open_scenes'], self.machines_info[index]['gpus'])))
            if result[0] == False:
                logger.error(f'打开场景失败，机器: {socket_client.address._host}:{socket_client.address._port}')
                raise Exception('打开场景失败')
            assert len(result[1]) == 2, '打开场景失败'
            print('waiting for airsim connection...')
            time.sleep(3 * len(self.machines_info[index]['open_scenes']) + 35)
            ip = result[1][0]
            ports = result[1][1]
            self.airsim_ip = ip
            self.airsim_ports = ports

            if isinstance(ip, bytes):
                ip = ip.decode('utf-8') 
            assert str(ip) == str(socket_client.address._host), '打开场景失败'
            assert len(ports) == len(self.machines_info[index]['open_scenes']), '打开场景失败'
            for i, port in enumerate(ports):
                if self.machines_info[index]['open_scenes'][i] is None:
                    self.airsim_clients[index][i] = None
                else:
                    self.airsim_clients[index][i] = airsim.MultirotorClient(ip=ip, port=port, timeout_value=airsim_timeout)
                    print(port)

            logger.info(f'打开场景完毕，机器{index}: {socket_client.address._host}:{socket_client.address._port}')
            return ports

        threads = []
        thread_results = []
        for index, socket_client in enumerate(socket_clients):
            threads.append(
                MyThread(_run_command, (index, socket_client))
            )
        for thread in threads:
            thread.setDaemon(True)
            thread.start()
        for thread in threads:
            thread.join()
        for thread in threads:
            thread.get_result()
            thread_results.append(thread.flag_ok)
        threads = []
        
        if not (np.array(thread_results) == True).all():
            raise Exception('打开场景失败')

        after = time.time()
        diff = after - before
        logger.info(f"启动时间：{diff}")

        assert self._confirmConnection(), 'server connect failed'

        self._closeSocketConnection()

        
    
    def collect_DDP(self, data_dir, workers):
        def init_worker(index, lock):
            with lock:
                multiprocessing.current_process().client_port = self.machines_info[0]['SOCKET_PORT']
                multiprocessing.current_process().machine_ip = self.machines_info[0]['MACHINE_IP']
                multiprocessing.current_process().client = self.airsim_clients[0][index.value]
                multiprocessing.current_process().port = self.airsim_ports[index.value]
                index.value += 1
        index = multiprocessing.Value('i', 0)
        lock = multiprocessing.Lock()
        with multiprocessing.Pool(workers, initializer=init_worker, initargs=(index, lock)) as p:
            r = list(tqdm.tqdm(p.imap_unordered(collect, data_dir), total=len(data_dir)))
    


    def _destroy_airsim_clients(self, kill_by_ports: list = None):
        """
        彻底销毁所有 AirSim 客户端连接，包括关闭 msgpackrpc socket。
        如果 kill_by_ports 指定端口列表（例如 [30001, 30002]），会杀掉所有监听这些端口的进程。
        """
        if not hasattr(self, "airsim_clients"):
            logger.warning("没有需要销毁的 AirSim 客户端")
            return

        for row in self.airsim_clients:
            for client in row:
                if client is None:
                    continue

                try:
                    # 1. 释放控制权
                    client.enableApiControl(False)
                    logger.info("已释放无人机 API 控制权")
                except Exception as e:
                    logger.warning(f"释放 API 控制权失败: {e}")

                try:
                    # 2. 重置仿真
                    client.reset()
                    logger.info("已重置仿真状态")
                except Exception as e:
                    logger.warning(f"重置仿真失败: {e}")

                try:
                    # 3. 关闭 msgpackrpc 连接（关键！）
                    if hasattr(client, 'client') and client.client is not None:
                        client.client.close()
                        logger.info("已关闭 msgpackrpc TCP 连接")
                except Exception as e:
                    logger.warning(f"关闭 msgpackrpc 连接失败: {e}")

                # 4. 解除对象引用
                client = None

        # 清空列表
        self.airsim_clients = []
        logger.info("所有 AirSim 客户端连接已销毁")

        # 如果需要按端口列表杀进程
        if kill_by_ports and isinstance(kill_by_ports, list):
            import subprocess
            import os
            import signal

            for port in kill_by_ports:
                # 使用 lsof 找到监听该端口的进程 PID
                result = subprocess.run(f"lsof -ti tcp:{port}", shell=True, capture_output=True, text=True)
                pids = result.stdout.strip().split()

                for pid in pids:
                    try:
                        os.kill(int(pid), signal.SIGKILL)
                        logger.info(f"已杀掉监听端口 {port} 的进程 PID: {pid}")
                    except Exception as e:
                        logger.warning(f"杀掉进程 PID {pid} 失败: {e}")

    def closeScenes(self):
        try:
            socket_clients = []
            for index, item in enumerate(self.machines_info):
                socket_clients.append(
                    msgpackrpc.Client(msgpackrpc.Address(item['MACHINE_IP'], item['SOCKET_PORT']), timeout=300)
                )

            for socket_client in socket_clients:
                if not self._confirmSocketConnection(socket_client):
                    logger.error('cannot establish socket')
                    raise Exception('cannot establish socket')

            self.socket_clients = socket_clients

            self._closeConnection()
            def _run_command(index, socket_client: msgpackrpc.Client):
                logger.info(f'开始关闭所有场景，机器{index}: {socket_client.address._host}:{socket_client.address._port}')
                result = socket_client.call('close_scenes', socket_client.address._host)
                logger.info(f'关闭所有场景完毕，机器{index}: {socket_client.address._host}:{socket_client.address._port}')
                return

            threads = []
            for index, socket_client in enumerate(socket_clients):
                threads.append(
                    MyThread(_run_command, (index, socket_client))
                )
            for thread in threads:
                thread.setDaemon(True)
                thread.start()
            for thread in threads:
                thread.join()
            threads = []

            self._closeSocketConnection()
        except Exception as e:
            logger.error(e)


    def move_path_by_waypoints_qwenvl(self, waypoints_list):
        velocity = 1
        drivetrain = airsim.DrivetrainType.ForwardOnly
        yaw_mode=airsim.YawMode(is_rate=False)
        lookahead=3
        adaptive_lookahead=1


        def move_path(airsim_client: airsim.VehicleClient, waypoints):
            results = []
            state_sensor = State(airsim_client, )
            imu_sensor = Imu(airsim_client, imu_name='Imu')
            
            path = [airsim.Vector3r(*waypoint[0:3]) for waypoint in waypoints]
            airsim_client.enableApiControl(True)
            airsim_client.armDisarm(True)
            airsim_client.simPause(False)
            state_info = state_sensor.retrieve()
            airsim_client.moveOnPathAsync(path=path, 
                                velocity=velocity, 
                                drivetrain=drivetrain, 
                                yaw_mode=yaw_mode, 
                                lookahead=lookahead, 
                                adaptive_lookahead=adaptive_lookahead)
            target_idx = 5
            current_idx = 0
            pos_queue = deque(maxlen=20)
            start_time = time.perf_counter()
            collision = False
            distance = 10000
            while True:
                time.sleep(0.005)
                if time.perf_counter() - start_time > 5:
                    collision = True
                    break
                target = path[current_idx]
                state_info = copy.deepcopy(state_sensor.retrieve())
                imu_info = copy.deepcopy(imu_sensor.retrieve())
                position = np.array(state_info['position'])
                pos_queue.append(position)
                if len(pos_queue) == pos_queue.maxlen:
                    recent_loc = position
                    history_loc = pos_queue.popleft()
                    delta_distance = np.linalg.norm(history_loc -recent_loc)
                    if delta_distance < 0.1:
                        print('move on path api: stuck max len')
                        collision = True
                        break
                new_distance = np.linalg.norm(position - np.array([target.x_val, target.y_val, target.z_val]))
                if new_distance > distance:
                    current_idx += 1
                    if current_idx == target_idx:
                        # airsim_client.simPause(True)
                        airsim_client.hoverAsync().join()
                        break
                    else:
                        distance = 10000
                else:
                    distance = new_distance
            
            results.append({'sensors': {'state': state_info, 'imu': imu_info}})
            return {'states': results, 'collision': collision}
        
        threads = []
        thread_results = []
        for index_1 in range(len(self.airsim_clients)):
            threads.append([])
            for index_2 in range(len(self.airsim_clients[index_1])):
                threads[index_1].append(
                    MyThread(move_path, (self.airsim_clients[index_1][index_2], waypoints_list[index_1][index_2]))
                )
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].setDaemon(True)
                threads[index_1][index_2].start()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].join()

        result_poses_list = []
        error_flag = False
        for index_1, _ in enumerate(threads):
            result_poses_list.append([])
            for index_2, _ in enumerate(threads[index_1]):
                result =  threads[index_1][index_2].get_result()
                result_poses_list[index_1].append(
                    result
                )
                if result is None:
                    error_flag = True
                thread_results.append(threads[index_1][index_2].flag_ok)
        threads = []
        if not (np.array(thread_results) == True).all():
            logger.error('move path by waypoints failed.')
            return None
        if error_flag:
            return None
        return result_poses_list
  

    
    def setPoses(self, poses: list) -> bool:
        def _setPoses(airsim_client: airsim.VehicleClient, pose: airsim.Pose) -> None:
            if airsim_client is None:
                raise Exception('error')
                return

            airsim_client.simSetKinematics(
                state=pose,
                ignore_collision=True,
            )
            airsim_client.simContinueForFrames(1)
            airsim_client.simPause(True)

            return

        threads = []
        thread_results = []
        for index_1 in range(len(self.airsim_clients)):
            threads.append([])
            for index_2 in range(len(self.airsim_clients[index_1])):
                threads[index_1].append(
                    MyThread(_setPoses, (self.airsim_clients[index_1][index_2], poses[index_1][index_2]))
                )
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].setDaemon(True)
                threads[index_1][index_2].start()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].join()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].get_result()
                thread_results.append(threads[index_1][index_2].flag_ok)
        threads = []
        if not (np.array(thread_results) == True).all():
            logger.error('setPoses失败')
            return False

        return True
    
    def setObjects(self, object_list: list):
        def _setObject(airsim_client: airsim.VehicleClient, object_info: dict) -> None:
            if airsim_client is None:
                raise Exception('error')
                return
            asset_name = object_info['asset_name']
            pose = object_info['pose']
            scale = object_info['scale']
            object_cnt = object_info['object_cnt']
            if object_cnt > 0:
                airsim_client.simDestroyObject('my_object_' + str(object_cnt - 1))
            success = airsim_client.simSpawnObject(
                    'my_object_' + str(object_cnt), asset_name, pose, scale, physics_enabled=False, is_blueprint=False)
            airsim_client.simContinueForFrames(1)
            airsim_client.simPause(True)
            return success

        threads = []
        thread_results = []
        cnt = 0
        for index_1 in range(len(self.airsim_clients)):
            threads.append([])
            for index_2 in range(len(self.airsim_clients[index_1])):
                object_list[cnt]['object_cnt'] = self.objects_name_cnt[index_1][index_2]
                threads[index_1].append(
                    MyThread(_setObject, (self.airsim_clients[index_1][index_2], object_list[cnt]))
                )
                self.objects_name_cnt[index_1][index_2] += 1
                cnt += 1

        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].setDaemon(True)
                threads[index_1][index_2].start()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].join()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].get_result()
                thread_results.append(threads[index_1][index_2].flag_ok)
        threads = []
        if not (np.array(thread_results) == True).all():
            logger.error('set Object失败')
            return False
        return True
    
    def getImageResponses(self, cameras=['FrontCamera', 'DownCamera'], poses=None):
        def _getImages(airsim_client: airsim.VehicleClient):
            if airsim_client is None:
                raise Exception('client is None.')
                return None, None
            time_sleep_cnt = 0
            while True:
                try:
                    ImageRequest = []
                    for camera_name in cameras:
                        ImageRequest.append(airsim.ImageRequest(camera_name, airsim.ImageType.Scene, pixels_as_float=False, compress=False))
                        ImageRequest.append(airsim.ImageRequest(camera_name, airsim.ImageType.DepthPerspective, pixels_as_float=True, compress=False))
                    image_datas = airsim_client.simGetImages(requests=ImageRequest)

                    images, depth_images = [], []
                    for idx, camera_name in enumerate(cameras):
                        rgb_resp = image_datas[2 * idx]
                        image = np.frombuffer(rgb_resp.image_data_uint8, dtype=np.uint8).reshape(rgb_resp.height, rgb_resp.width, 3)
                        depth_resp = image_datas[2* idx + 1]
                        depth_img_in_meters = airsim.list_to_2d_float_array(depth_resp.image_data_float, depth_resp.width, depth_resp.height)
                        depth_image = (np.clip(depth_img_in_meters, 0, 100) / 100 * 255).astype(np.uint8)
                        images.append(image)
                        depth_images.append(depth_image)
                    break
                except Exception as e:
                    time_sleep_cnt += 1
                    logger.error("图片获取错误: " + str(e))
                    logger.error('time_sleep_cnt: {}'.format(time_sleep_cnt))
                    time.sleep(1)
                if time_sleep_cnt > 10:
                    raise Exception('图片获取失败')
            return images, depth_images

        threads = []
        thread_results = []
        for index_1 in range(len(self.airsim_clients)):
            threads.append([])
            for index_2 in range(len(self.airsim_clients[index_1])):
                threads[index_1].append(
                    MyThread(_getImages, (self.airsim_clients[index_1][index_2], ))
                )
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].setDaemon(True)
                threads[index_1][index_2].start()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].join()
        responses = []
        for index_1, _ in enumerate(threads):
            responses.append([])
            for index_2, _ in enumerate(threads[index_1]):
                responses[index_1].append(
                    threads[index_1][index_2].get_result()
                )
                thread_results.append(threads[index_1][index_2].flag_ok)
        threads = []
        if not (np.array(thread_results) == True).all():
            logger.error('getImageResponses失败')
            return None
        return responses


    def getImageResponsesForRecord(self, cameras=['FrontCameraRecord', 'DownCameraRecord'], poses=None):
        def _getImages(airsim_client: airsim.VehicleClient):
            if airsim_client is None:
                raise Exception('client is None.')
                return None, None
            time_sleep_cnt = 0
            while True:
                try:

                    ImageRequest = []
                    for camera_name in cameras:
                        ImageRequest.append(airsim.ImageRequest(camera_name, airsim.ImageType.Scene, pixels_as_float=False, compress=False))
                        ImageRequest.append(airsim.ImageRequest(camera_name, airsim.ImageType.DepthPerspective, pixels_as_float=True, compress=False))

                    image_datas = airsim_client.simGetImages(requests=ImageRequest)

                    images, depth_images = [], []
                    for idx, camera_name in enumerate(cameras):
                        rgb_resp = image_datas[2 * idx]
                        image = np.frombuffer(rgb_resp.image_data_uint8, dtype=np.uint8).reshape(rgb_resp.height, rgb_resp.width, 3)
                        depth_resp = image_datas[2* idx + 1]
                        depth_img_in_meters = airsim.list_to_2d_float_array(depth_resp.image_data_float, depth_resp.width, depth_resp.height)
                        depth_image = (np.clip(depth_img_in_meters, 0, 100) / 100 * 255).astype(np.uint8)
                        images.append(image)
                        depth_images.append(depth_image)

                    break
                except Exception as e:
                    time_sleep_cnt += 1
                    logger.error("图片获取错误: " + str(e))
                    logger.error('time_sleep_cnt: {}'.format(time_sleep_cnt))
                    time.sleep(1)
                if time_sleep_cnt > 10:
                    raise Exception('图片获取失败')
            return images, depth_images

        threads = []
        thread_results = []
        for index_1 in range(len(self.airsim_clients)):
            threads.append([])
            for index_2 in range(len(self.airsim_clients[index_1])):
                threads[index_1].append(
                    MyThread(_getImages, (self.airsim_clients[index_1][index_2], ))
                )
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].setDaemon(True)
                threads[index_1][index_2].start()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].join()
        responses = []
        for index_1, _ in enumerate(threads):
            responses.append([])
            for index_2, _ in enumerate(threads[index_1]):
                responses[index_1].append(
                    threads[index_1][index_2].get_result()
                )
                thread_results.append(threads[index_1][index_2].flag_ok)
        threads = []
        if not (np.array(thread_results) == True).all():
            logger.error('getImageResponses失败')
            return None

        return responses

    def getSensorInfo(self, ):
        def get_sensor_info(airsim_client: airsim.VehicleClient, ):
            state_sensor = State(airsim_client, )
            imu_sensor = Imu(airsim_client)
            state_info = state_sensor.retrieve()
            imu_info = imu_sensor.retrieve()
            return {'sensors': {'state':state_info, 'imu': imu_info}}
        threads = []
        thread_results = []
        for index_1 in range(len(self.airsim_clients)):
            threads.append([])
            for index_2 in range(len(self.airsim_clients[index_1])):
                threads[index_1].append(
                    MyThread(get_sensor_info, (self.airsim_clients[index_1][index_2], ))
                )
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].setDaemon(True)
                threads[index_1][index_2].start()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].join()

        results = []
        for index_1, _ in enumerate(threads):
            results.append([])
            for index_2, _ in enumerate(threads[index_1]):
                results[index_1].append(
                    threads[index_1][index_2].get_result()
                )
                thread_results.append(threads[index_1][index_2].flag_ok)
        threads = []
        if not (np.array(thread_results) == True).all():
            logger.error('getSensorInfo failed.')
            return None
        return results 


