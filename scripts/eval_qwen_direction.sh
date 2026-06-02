

root_dir=/opt/data/private/lwt-project/3DG-VLN

SIM_ROOT=/opt/data/private/lwt-project/FOV-dataset/env_unzip
SIM_PORT=30000
MASTER_PORT=60001
GPU_ID=0

MAX_RETRY=-1
SLEEP_SECONDS=10

retry_count=0


kill_port() {
    port=$1
    echo "[INFO] Killing processes on port ${port}..."

    pids=$(lsof -ti:${port})

    if [ -n "$pids" ]; then
        echo "[INFO] Found PIDs on port ${port}: ${pids}"
        kill -9 ${pids}
        sleep 1
    else
        echo "[INFO] No process found on port ${port}"
    fi
}


prepare_simulator() {
    echo "=========================================="
    echo "[INFO] Prepare simulator at $(date)"
    echo "=========================================="

    kill_port ${SIM_PORT}
    kill_port $((SIM_PORT + 1))
    kill_port $((SIM_PORT + 2))

    echo "[INFO] Starting AirVLN simulator server on port ${SIM_PORT}..."


    (
        cd /opt/data/private/lwt-project/3DG-VLN/airsim_plugin || exit 1

        nohup python AirVLNSimulatorServerTool.py \
            --port ${SIM_PORT} \
            --root_path ${SIM_ROOT} \
            > simulator_${SIM_PORT}.log 2>&1 &

        sim_pid=$!
        echo "[INFO] Simulator PID: ${sim_pid}"
    )


    sleep 10
}


while true
do
    prepare_simulator

    echo "=========================================="
    echo "[INFO] Start eval at $(date)"
    echo "[INFO] Retry count: ${retry_count}"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES=${GPU_ID} python -u $root_dir/src/vlnce_src/eval_qwenvl_direction.py \
        --run_type eval \
        --name  qwen2_5vl \
        --gpu_id 0 \
        --simulator_tool_port ${SIM_PORT} \
        --DDP_MASTER_PORT ${MASTER_PORT} \
        --batchSize 2 \
        --always_help True \
        --use_gt True \
        --maxWaypoints 100 \
        --use_target_des True \
        --dataset_path /test/ \
        --eval_save_path /result/test \
        --model_path /model/3DG-VLN \
        --map_spawn_area_json_path /meta/map_spawnarea_info.json \
        --object_name_json_path /meta/object_description.json \
        --groundingdino_config $root_dir/src/model_wrapper/utils/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
        --groundingdino_model_path $root_dir/src/model_wrapper/utils/GroundingDINO/groundingdino_swint_ogc.pth \
        --target_des_path '/meta/object_attribute' \
        --obj_desc_json_path '/meta/instructions.json'


    exit_code=$?

    echo "[INFO] Eval exited with code: ${exit_code}"
    echo "[INFO] End time: $(date)"

    if [ ${exit_code} -eq 0 ]; then
        echo "[INFO] Eval finished normally. Stop restarting."
        break
    fi

    if [ ${exit_code} -eq 130 ]; then
        echo "[INFO] Interrupted by Ctrl+C. Stop restarting."
        break
    fi

    retry_count=$((retry_count + 1))

    if [ ${MAX_RETRY} -ne -1 ] && [ ${retry_count} -ge ${MAX_RETRY} ]; then
        echo "[ERROR] Reached max retry count: ${MAX_RETRY}. Stop restarting."
        break
    fi

    echo "[WARNING] Eval crashed abnormally. Restart after ${SLEEP_SECONDS} seconds..."
    sleep ${SLEEP_SECONDS}
done
