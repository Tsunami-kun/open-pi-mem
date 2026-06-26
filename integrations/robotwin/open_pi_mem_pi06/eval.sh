#!/bin/bash
set -euo pipefail

if [ "$#" -ne 6 ]; then
    echo "usage: bash eval.sh <task_name> <task_config> <train_config_name> <model_name> <seed> <gpu_id>" >&2
    exit 2
fi

export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.4}

policy_name=open_pi_mem_pi06
task_name=${1}
task_config=${2}
train_config_name=${3}
model_name=${4}
seed=${5}
gpu_id=${6}

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../..

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/${policy_name}/deploy_policy.yml \
    --overrides \
    --task_name "${task_name}" \
    --task_config "${task_config}" \
    --train_config_name "${train_config_name}" \
    --model_name "${model_name}" \
    --ckpt_setting "${model_name}" \
    --seed "${seed}" \
    --policy_name "${policy_name}"
