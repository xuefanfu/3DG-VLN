import argparse
import copy
import json
import os

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate SR, OSR, NE, and SPL for UAV navigation results."
    )

    parser.add_argument(
        "--eval_save_path",
        type=str,
        required=True,
        help="Directory containing the evaluation trajectory results.",
    )
    parser.add_argument(
        "--eval_test_path",
        type=str,
        required=True,
        help="Path to the test dataset.",
    )
    parser.add_argument(
        "--eval_unscene_path",
        type=str,
        required=True,
        help="Path to the unseen-scene dataset.",
    )
    parser.add_argument(
        "--eval_unobject_path",
        type=str,
        required=True,
        help="Path to the unseen-object dataset.",
    )
    parser.add_argument(
        "--object_info_path",
        type=str,
        required=True,
        help="Path to map_spawnarea_info.json.",
    )

    return parser.parse_args()


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


def add_trajectory_scenes(dataset_path, traj_scene):
    """Add trajectory-name to scene-name mappings from one dataset split."""
    for scene_name in os.listdir(dataset_path):
        scene_path = os.path.join(dataset_path, scene_name)

        if not os.path.isdir(scene_path):
            continue

        for traj_name in os.listdir(scene_path):
            traj_path = os.path.join(scene_path, traj_name)

            if os.path.isdir(traj_path):
                traj_scene[traj_name] = scene_name


def validate_paths(args):
    directory_paths = {
        "eval_save_path": args.eval_save_path,
        "eval_test_path": args.eval_test_path,
        "eval_unscene_path": args.eval_unscene_path,
        "eval_unobject_path": args.eval_unobject_path,
    }

    for name, path in directory_paths.items():
        if not os.path.isdir(path):
            raise FileNotFoundError(f"{name} does not exist or is not a directory: {path}")

    if not os.path.isfile(args.object_info_path):
        raise FileNotFoundError(
            f"object_info_path does not exist or is not a file: "
            f"{args.object_info_path}"
        )


def main():
    args = parse_args()
    validate_paths(args)

    eval_save_path = args.eval_save_path
    eval_test_path = args.eval_test_path
    eval_unscene_path = args.eval_unscene_path
    eval_unobject_path = args.eval_unobject_path
    object_info_path = args.object_info_path

    print(f"Evaluation results: {eval_save_path}")
    print(f"Test dataset:       {eval_test_path}")
    print(f"Unseen scenes:      {eval_unscene_path}")
    print(f"Unseen objects:     {eval_unobject_path}")
    print(f"Object information: {object_info_path}")
    print("*******************************************************************************")

    with open(object_info_path, "r", encoding="utf-8") as f:
        map_area_dict = json.load(f)

    # 构建“轨迹名称 -> 场景名称”的映射
    traj_scene = {}
    add_trajectory_scenes(eval_test_path, traj_scene)
    add_trajectory_scenes(eval_unscene_path, traj_scene)
    add_trajectory_scenes(eval_unobject_path, traj_scene)

    # 统计指标
    oracle_success = 0
    oracle_success_traj = []
    distance_list = []
    success_traj = []
    spl_list = []

    traj_names = [
        name
        for name in os.listdir(eval_save_path)
        if os.path.isdir(os.path.join(eval_save_path, name))
    ]

    if not traj_names:
        raise RuntimeError(f"No trajectory directories found in: {eval_save_path}")

    for traj_name in traj_names:
        ori_traj_name = copy.deepcopy(traj_name)

        if "success_" in traj_name:
            success_traj.append(traj_name)

        traj_path = os.path.join(eval_save_path, traj_name)
        traj_log_path = os.path.join(traj_path, "log")

        ori_info_path = os.path.join(traj_path, "ori_info.json")
        with open(ori_info_path, "r", encoding="utf-8") as f:
            traj_ori_path = json.load(f)["ori_traj_dir"]

        mark_path = os.path.join(traj_ori_path, "mark.json")
        with open(mark_path, "r", encoding="utf-8") as f:
            traj_object_position = json.load(f)["target"]["position"]

        clean_traj_name = traj_name.replace("success_", "").replace("oracle_", "")

        if clean_traj_name not in traj_scene:
            raise KeyError(
                f"Trajectory '{clean_traj_name}' was not found in test, "
                "unscene, or unobject datasets."
            )

        scene_name = traj_scene[clean_traj_name]

        if scene_name not in map_area_dict:
            raise KeyError(
                f"Scene '{scene_name}' was not found in object information file."
            )

        _, closest_area_info = find_closest_area(
            traj_object_position,
            map_area_dict[scene_name],
        )

        if closest_area_info is None:
            raise RuntimeError(
                f"No valid target area was found for trajectory: {clean_traj_name}"
            )

        object_position = [
            closest_area_info[9],
            closest_area_info[10],
            closest_area_info[11],
        ]

        # 读取所有 JSON 日志，并根据数字文件名排序
        log_files = [
            name
            for name in os.listdir(traj_log_path)
            if name.endswith(".json")
        ]
        log_files.sort(key=lambda x: int(os.path.splitext(x)[0]))

        if not log_files:
            raise RuntimeError(f"No JSON log files found in: {traj_log_path}")

        # 计算 Oracle Success
        for log_name in log_files:
            log_path = os.path.join(traj_log_path, log_name)

            with open(log_path, "r", encoding="utf-8") as f:
                log_data = json.load(f)

            log_position = log_data["sensors"]["state"]["position"]
            distance = np.linalg.norm(
                np.array(log_position) - np.array(object_position)
            )

            if distance <= 10:
                oracle_success += 1
                oracle_success_traj.append(clean_traj_name)
                break

        # 计算 NE
        last_log_path = os.path.join(traj_log_path, log_files[-1])

        with open(last_log_path, "r", encoding="utf-8") as f:
            log_data = json.load(f)

        log_position = log_data["sensors"]["state"]["position"]
        distance = np.linalg.norm(
            np.array(log_position) - np.array(object_position)
        )

        if distance <= 500:
            distance_list.append(distance)

        # 计算 SPL
        if "success_" in ori_traj_name:
            pred_length = 0.0
            pre_point = None

            for log_name in log_files:
                log_path = os.path.join(traj_log_path, log_name)

                with open(log_path, "r", encoding="utf-8") as f:
                    log_data = json.load(f)

                point = log_data["sensors"]["state"]["position"]

                if pre_point is not None:
                    pred_length += np.linalg.norm(
                        np.array(pre_point) - np.array(point)
                    )

                pre_point = point

            ori_traj_path = os.path.join(traj_ori_path, "merged_data.json")

            with open(ori_traj_path, "r", encoding="utf-8") as f:
                ori_data = json.load(f)["trajectory_raw_detailed"]

            path_length = 0.0

            for i in range(len(ori_data) - 1):
                p1 = np.array(ori_data[i]["position"])
                p2 = np.array(ori_data[i + 1]["position"])
                path_length += np.linalg.norm(p2 - p1)

            path_length -= 10

            if max(path_length, pred_length) > 0:
                spl = path_length / max(path_length, pred_length)
            else:
                spl = 0.0

            spl_list.append(max(spl, 0.0))
        else:
            spl_list.append(0.0)

    # 汇总指标
    ne_mean = np.mean(distance_list) if distance_list else float("nan")
    oracle_success_rate = oracle_success / len(traj_names) * 100
    success_rate = len(success_traj) / len(traj_names) * 100
    avg_spl = np.mean(np.array(spl_list)) * 100

    print(f"ne_mean: {ne_mean:.2f}")
    print(f"oracle_success_rate: {oracle_success_rate:.2f}%")
    print(f"len(success_traj): {len(success_traj)}")
    print(f"success_rate: {success_rate:.2f}%")
    print(f"avg_spl: {avg_spl:.2f}%")
    print("*******************************************************************************")


if __name__ == "__main__":
    main()
