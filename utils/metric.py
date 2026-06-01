import copy
import json
import os

import numpy as np


def find_closest_area(coord, areas):
    def euclidean_distance(coord1, coord2):
        return np.sqrt(sum((np.array(coord1) - np.array(coord2)) ** 2))

    min_distance = float("inf")
    closest_area = None
    closest_area_info = None

    for area in areas:
        if len(area) < 18:
            continue

        true_area = [area[0] + 1, area[1] + 1, area[2] + 0.5]
        distance = euclidean_distance(coord, true_area)

        if distance < min_distance:
            min_distance = distance
            closest_area = true_area
            closest_area_info = area

    return closest_area, closest_area_info


def main():
    # 测试结果地址
    eavl_save_path = ""

    # 测试数据集地址
    eval_test_path = "./test"
    eval_unscene_path = "./unscene"
    eval_unobject_path = "./unobject"

    # 目标位置信息
    object_info_path = "./map_spawnarea_info.json"
    with open(object_info_path, "r") as f:
        map_area_dict = json.load(f)

    # 构建轨迹场景名称列表
    traj_scene = {}

    for scene_name in os.listdir(eval_test_path):
        scene_path = os.path.join(eval_test_path, scene_name)
        for traj_name in os.listdir(scene_path):
            traj_scene[traj_name] = scene_name

    for scene_name in os.listdir(eval_unscene_path):
        scene_path = os.path.join(eval_unscene_path, scene_name)
        for traj_name in os.listdir(scene_path):
            traj_scene[traj_name] = scene_name

    for scene_name in os.listdir(eval_unobject_path):
        scene_path = os.path.join(eval_unobject_path, scene_name)
        for traj_name in os.listdir(scene_path):
            traj_scene[traj_name] = scene_name

    # 统计 oracle_success
    oracle_success = 0
    oracle_success_traj = []
    distance_list = []
    success_traj = []
    spl_list = []

    traj_names = os.listdir(eavl_save_path)


    for traj_name in traj_names:
        ori_traj_name = copy.deepcopy(traj_name)

        if "success_" in traj_name:
            success_traj.append(traj_name)

        traj_path = os.path.join(eavl_save_path, traj_name)
        traj_log_path = os.path.join(traj_path, "log")

        ori_info_path = os.path.join(eavl_save_path, traj_name, "ori_info.json")
        with open(ori_info_path, "r") as f:
            traj_ori_path = json.load(f)["ori_traj_dir"]

        mark_path = os.path.join(traj_ori_path, "mark.json")
        with open(mark_path, "r") as f:
            traj_object_position = json.load(f)["target"]["position"]

        traj_name = traj_name.replace("success_", "").replace("oracle_", "")
        scene_name = traj_scene[traj_name]

        _, closest_area_info = find_closest_area(
            traj_object_position,
            map_area_dict[scene_name],
        )

        object_position = [
            closest_area_info[9],
            closest_area_info[10],
            closest_area_info[11],
        ]

        # 读取所有 log 文件，并排序
        log_files = os.listdir(traj_log_path)
        log_files.sort(key=lambda x: int(x.split(".")[0]))

        # 计算 oracle_success 数量
        for j, log_name in enumerate(log_files):
            log_path = os.path.join(traj_log_path, log_name)

            with open(log_path, "r") as f:
                log_data = json.load(f)

            log_position = log_data["sensors"]["state"]["position"]

            # 计算距离
            distance = np.linalg.norm(
                np.array(log_position) - np.array(object_position)
            )

            if distance <= 10:
                oracle_success += 1
                oracle_success_traj.append(traj_name)
                break

        # 计算 NE
        last_log_path = os.path.join(traj_log_path, log_files[-1])

        with open(last_log_path, "r") as f:
            log_data = json.load(f)

        log_position = log_data["sensors"]["state"]["position"]

        distance = np.linalg.norm(
            np.array(log_position) - np.array(object_position)
        )

        # distance = distance - 10
        if distance > 500:
            pass
        else:
            distance_list.append(distance)

        if "success_" in ori_traj_name:
            # 计算 SPL
            pred_length = 0
            pre_point = None

            for log_name in log_files:
                log_path = os.path.join(traj_log_path, log_name)

                with open(log_path, "r") as f:
                    log_data = json.load(f)

                point = log_data["sensors"]["state"]["position"]

                if pre_point is not None:
                    pred_length += np.linalg.norm(
                        np.array(pre_point) - np.array(point)
                    )

                pre_point = point

            ori_traj_path = os.path.join(traj_ori_path, "merged_data.json")

            with open(ori_traj_path, "r") as f:
                ori_data = json.load(f)["trajectory_raw_detailed"]

            path_length = 0

            for i in range(len(ori_data) - 1):
                p1 = np.array(ori_data[i]["position"])
                p2 = np.array(ori_data[i + 1]["position"])
                path_length += np.linalg.norm(p2 - p1)

            path_length -= 10

            spl = path_length / max(path_length, pred_length)
            spl = max(spl, 0)

            spl_list.append(spl)
        else:
            spl_list.append(0)

    # 计算 NE 的平均值
    ne_mean = np.mean(distance_list)
    print(f"ne_mean: {ne_mean:.2f}")

    # 计算 OSR
    oracle_success_rate = oracle_success / len(traj_names) * 100
    print(f"oracle_success_rate: {oracle_success_rate:.2f}%")

    # 计算成功率
    print("len(success_traj)", len(success_traj))

    success_rate = len(success_traj) / len(traj_names) * 100
    print(f"success_rate: {success_rate:.2f}%")

    # 计算 SPL
    avg_spl = np.mean(np.array(spl_list)) * 100
    print(f"avg_spl: {avg_spl:.2f}%")

    print("*******************************************************************************")


if __name__ == "__main__":
    main()
