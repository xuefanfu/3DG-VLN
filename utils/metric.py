import argparse
import json
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_SUCCESS_RADIUS = 10.0
DEFAULT_MAX_NE_DISTANCE = 500.0


def parse_args() -> argparse.Namespace:
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
    parser.add_argument(
        "--success_radius",
        type=float,
        default=DEFAULT_SUCCESS_RADIUS,
        help=f"Success radius in meters. Default: {DEFAULT_SUCCESS_RADIUS}.",
    )
    parser.add_argument(
        "--max_ne_distance",
        type=float,
        default=DEFAULT_MAX_NE_DISTANCE,
        help=(
            "Ignore terminal navigation errors larger than this threshold. "
            f"Default: {DEFAULT_MAX_NE_DISTANCE}."
        ),
    )

    return parser.parse_args()


def validate_paths(args: argparse.Namespace) -> None:
    directory_paths = {
        "eval_save_path": args.eval_save_path,
        "eval_test_path": args.eval_test_path,
        "eval_unscene_path": args.eval_unscene_path,
        "eval_unobject_path": args.eval_unobject_path,
    }

    for name, path in directory_paths.items():
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"{name} does not exist or is not a directory: {path}"
            )

    if not os.path.isfile(args.object_info_path):
        raise FileNotFoundError(
            "object_info_path does not exist or is not a file: "
            f"{args.object_info_path}"
        )

    if args.success_radius < 0:
        raise ValueError("success_radius must be non-negative.")

    if args.max_ne_distance <= 0:
        raise ValueError("max_ne_distance must be positive.")


def find_closest_area(
    coord: Sequence[float],
    areas: Sequence[Sequence[float]],
) -> Tuple[Optional[List[float]], Optional[Sequence[float]]]:
    """Find the target-area record closest to the annotated target position."""
    coord_array = np.asarray(coord, dtype=np.float64)

    min_distance = float("inf")
    closest_area = None
    closest_area_info = None

    for area in areas:
        if len(area) < 18:
            continue

        true_area = np.asarray(
            [area[0] + 1.0, area[1] + 1.0, area[2] + 0.5],
            dtype=np.float64,
        )
        distance = float(np.linalg.norm(coord_array - true_area))

        if distance < min_distance:
            min_distance = distance
            closest_area = true_area.tolist()
            closest_area_info = area

    return closest_area, closest_area_info


def add_trajectory_scenes(
    dataset_path: str,
    traj_scene: Dict[str, str],
) -> None:
    """Build a mapping from trajectory name to scene name."""
    for scene_name in os.listdir(dataset_path):
        scene_path = os.path.join(dataset_path, scene_name)

        if not os.path.isdir(scene_path):
            continue

        for traj_name in os.listdir(scene_path):
            traj_path = os.path.join(scene_path, traj_name)

            if os.path.isdir(traj_path):
                traj_scene[traj_name] = scene_name


def remove_result_prefixes(traj_name: str) -> str:
    """Remove result-state prefixes such as success_ and oracle_."""
    clean_name = traj_name

    changed = True
    while changed:
        changed = False
        for prefix in ("success_", "oracle_"):
            if clean_name.startswith(prefix):
                clean_name = clean_name[len(prefix) :]
                changed = True

    return clean_name


def log_sort_key(filename: str):
    """Sort numeric log names such as 0.json, 1.json, ..., 10.json."""
    stem = os.path.splitext(filename)[0]

    if stem.isdigit():
        return 0, int(stem)

    numbers = re.findall(r"\d+", stem)
    if numbers:
        return 1, int(numbers[-1]), stem

    return 2, stem


def get_sorted_log_files(log_dir: str) -> List[str]:
    if not os.path.isdir(log_dir):
        raise FileNotFoundError(f"Log directory does not exist: {log_dir}")

    log_files = [
        filename
        for filename in os.listdir(log_dir)
        if filename.lower().endswith(".json")
        and os.path.isfile(os.path.join(log_dir, filename))
    ]
    log_files.sort(key=log_sort_key)

    if not log_files:
        raise RuntimeError(f"No JSON log files found in: {log_dir}")

    return log_files


def extract_position(log_data: dict, log_path: str) -> np.ndarray:
    try:
        position = log_data["sensors"]["state"]["position"]
    except (KeyError, TypeError) as exc:
        raise KeyError(
            "Missing log position field "
            f"['sensors']['state']['position'] in: {log_path}"
        ) from exc

    position_array = np.asarray(position, dtype=np.float64)

    if position_array.shape != (3,):
        raise ValueError(
            f"Position must contain exactly three values in {log_path}, "
            f"but got shape {position_array.shape}."
        )

    return position_array


def load_positions_from_logs(log_dir: str) -> List[np.ndarray]:
    """Read UAV positions from all JSON files in a log directory."""
    positions = []

    for log_name in get_sorted_log_files(log_dir):
        log_path = os.path.join(log_dir, log_name)

        with open(log_path, "r", encoding="utf-8") as f:
            log_data = json.load(f)

        positions.append(extract_position(log_data, log_path))

    return positions


def calculate_path_length(positions: Sequence[np.ndarray]) -> float:
    """Calculate the accumulated Euclidean length of a position sequence."""
    if len(positions) < 2:
        return 0.0

    return float(
        sum(
            np.linalg.norm(current - previous)
            for previous, current in zip(positions[:-1], positions[1:])
        )
    )


def get_reference_path_length(
    traj_ori_path: str,
    pred_positions: Sequence[np.ndarray],
    object_position: np.ndarray,
    success_radius: float,
    clean_traj_name: str,
) -> float:
    """
    Calculate the reference path length L*.

    Priority:
    1. Use the original trajectory logs in ``traj_ori_path/log``.
    2. If those logs do not exist, use the straight-line distance from the
       predicted trajectory's first position to the target success region.

    The subtraction of ``success_radius`` preserves the original metric logic,
    which measures the path required to enter the success region rather than
    to reach the target center.
    """
    ori_log_dir = os.path.join(traj_ori_path, "log")

    if os.path.isdir(ori_log_dir):
        try:
            ori_positions = load_positions_from_logs(ori_log_dir)
        except (FileNotFoundError, RuntimeError, KeyError, ValueError) as exc:
            print(
                f"[WARN] Cannot use original logs for {clean_traj_name}: {exc}"
            )
        else:
            if len(ori_positions) >= 2:
                original_length = calculate_path_length(ori_positions)
                return max(original_length - success_radius, 0.0)

            print(
                f"[WARN] Original log has fewer than two positions for "
                f"{clean_traj_name}: {ori_log_dir}"
            )
    else:
        print(
            f"[WARN] Original log directory is missing for {clean_traj_name}: "
            f"{ori_log_dir}"
        )

    if not pred_positions:
        raise RuntimeError(
            f"Cannot compute fallback reference length for {clean_traj_name}: "
            "the predicted log contains no positions."
        )

    straight_distance = float(
        np.linalg.norm(pred_positions[0] - object_position)
    )
    reference_length = max(straight_distance - success_radius, 0.0)

    print(
        f"[WARN] Use straight-line reference length for {clean_traj_name}: "
        f"{reference_length:.3f} m"
    )

    return reference_length


def calculate_spl(
    success: bool,
    reference_length: float,
    predicted_length: float,
) -> float:
    """Calculate per-trajectory SPL: S * L* / max(L*, P)."""
    if not success:
        return 0.0

    denominator = max(reference_length, predicted_length)

    if denominator <= 1e-8:
        # The UAV starts inside the success region and does not move.
        return 1.0

    return float(reference_length / denominator)


def main() -> None:
    args = parse_args()
    validate_paths(args)

    print(f"Evaluation results: {args.eval_save_path}")
    print(f"Test dataset:       {args.eval_test_path}")
    print(f"Unseen scenes:      {args.eval_unscene_path}")
    print(f"Unseen objects:     {args.eval_unobject_path}")
    print(f"Object information: {args.object_info_path}")
    print("*******************************************************************************")

    with open(args.object_info_path, "r", encoding="utf-8") as f:
        map_area_dict = json.load(f)

    traj_scene: Dict[str, str] = {}
    add_trajectory_scenes(args.eval_test_path, traj_scene)
    add_trajectory_scenes(args.eval_unscene_path, traj_scene)
    add_trajectory_scenes(args.eval_unobject_path, traj_scene)

    oracle_success_count = 0
    terminal_distances: List[float] = []
    successful_trajectories: List[str] = []
    spl_values: List[float] = []

    traj_names = sorted(
        name
        for name in os.listdir(args.eval_save_path)
        if os.path.isdir(os.path.join(args.eval_save_path, name))
    )

    if not traj_names:
        raise RuntimeError(
            f"No trajectory directories found in: {args.eval_save_path}"
        )

    for result_traj_name in traj_names:
        result_traj_path = os.path.join(
            args.eval_save_path,
            result_traj_name,
        )
        pred_log_dir = os.path.join(result_traj_path, "log")

        success = result_traj_name.startswith("success_")
        clean_traj_name = remove_result_prefixes(result_traj_name)

        if success:
            successful_trajectories.append(clean_traj_name)

        ori_info_path = os.path.join(result_traj_path, "ori_info.json")
        if not os.path.isfile(ori_info_path):
            raise FileNotFoundError(
                f"ori_info.json does not exist: {ori_info_path}"
            )

        with open(ori_info_path, "r", encoding="utf-8") as f:
            ori_info = json.load(f)

        if "ori_traj_dir" not in ori_info:
            raise KeyError(
                f"Missing 'ori_traj_dir' in: {ori_info_path}"
            )

        traj_ori_path = ori_info["ori_traj_dir"]
        mark_path = os.path.join(traj_ori_path, "mark.json")

        if not os.path.isfile(mark_path):
            raise FileNotFoundError(f"mark.json does not exist: {mark_path}")

        with open(mark_path, "r", encoding="utf-8") as f:
            mark_data = json.load(f)

        try:
            traj_object_position = mark_data["target"]["position"]
        except (KeyError, TypeError) as exc:
            raise KeyError(
                f"Missing ['target']['position'] in: {mark_path}"
            ) from exc

        if clean_traj_name not in traj_scene:
            raise KeyError(
                f"Trajectory '{clean_traj_name}' was not found in test, "
                "unscene, or unobject datasets."
            )

        scene_name = traj_scene[clean_traj_name]

        if scene_name not in map_area_dict:
            raise KeyError(
                f"Scene '{scene_name}' was not found in "
                f"{args.object_info_path}."
            )

        _, closest_area_info = find_closest_area(
            traj_object_position,
            map_area_dict[scene_name],
        )

        if closest_area_info is None:
            raise RuntimeError(
                f"No valid target area found for trajectory: {clean_traj_name}"
            )

        object_position = np.asarray(
            [
                closest_area_info[9],
                closest_area_info[10],
                closest_area_info[11],
            ],
            dtype=np.float64,
        )

        pred_positions = load_positions_from_logs(pred_log_dir)
        distances_to_target = [
            float(np.linalg.norm(position - object_position))
            for position in pred_positions
        ]

        # OSR: any point in the trajectory enters the success radius.
        if any(
            distance <= args.success_radius
            for distance in distances_to_target
        ):
            oracle_success_count += 1

        # NE: distance between the final UAV position and target position.
        final_distance = distances_to_target[-1]
        if final_distance <= args.max_ne_distance:
            terminal_distances.append(final_distance)


        # SPL: failed trajectories contribute zero.
        if success:
            predicted_length = calculate_path_length(pred_positions)
            reference_length = get_reference_path_length(
                traj_ori_path=traj_ori_path,
                pred_positions=pred_positions,
                object_position=object_position,
                success_radius=args.success_radius,
                clean_traj_name=clean_traj_name,
            )
            spl = calculate_spl(
                success=True,
                reference_length=reference_length,
                predicted_length=predicted_length,
            )
            spl_values.append(spl)
        else:
            spl_values.append(0.0)

    trajectory_count = len(traj_names)

    ne_mean = (
        float(np.mean(terminal_distances))
        if terminal_distances
        else float("nan")
    )
    oracle_success_rate = oracle_success_count / trajectory_count * 100.0
    success_rate = len(successful_trajectories) / trajectory_count * 100.0
    average_spl = float(np.mean(spl_values)) * 100.0

    print(f"num_trajectories: {trajectory_count}")
    print(f"ne_mean: {ne_mean:.2f}")
    print(f"oracle_success_rate: {oracle_success_rate:.2f}%")
    print(f"len(success_traj): {len(successful_trajectories)}")
    print(f"success_rate: {success_rate:.2f}%")
    print(f"avg_spl: {average_spl:.2f}%")
    print("*******************************************************************************")


if __name__ == "__main__":
    main()
