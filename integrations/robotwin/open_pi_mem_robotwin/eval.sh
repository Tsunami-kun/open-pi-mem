#!/bin/bash
set -euo pipefail

if [ "$#" -ne 5 ]; then
    echo "usage: bash eval.sh <task_name> <task_config> <checkpoint_path> <seed> <gpu_id>" >&2
    exit 2
fi

policy_name=open_pi_mem_robotwin
task_name=${1}
task_config=${2}
checkpoint_path=${3}
seed=${4}
gpu_id=${5}

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../..

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/${policy_name}/deploy_policy.yml \
    --overrides \
    --task_name "${task_name}" \
    --task_config "${task_config}" \
    --checkpoint_path "${checkpoint_path}" \
    --ckpt_setting "${checkpoint_path}" \
    --seed "${seed}" \
    --policy_name "${policy_name}"
