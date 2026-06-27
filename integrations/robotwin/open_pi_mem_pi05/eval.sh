#!/bin/bash
set -euo pipefail

if [ "$#" -ne 8 ]; then
    echo "usage: bash eval.sh <task_name> <task_config> <train_config_name> <model_name> <checkpoint_id> <seed> <gpu_id> <policy_port>" >&2
    exit 2
fi

policy_name=open_pi_mem_pi05
task_name=${1}
task_config=${2}
train_config_name=${3}
model_name=${4}
checkpoint_id=${5}
seed=${6}
gpu_id=${7}
policy_port=${8}

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
    --checkpoint_id "${checkpoint_id}" \
    --ckpt_setting "${model_name}" \
    --seed "${seed}" \
    --policy_name "${policy_name}" \
    --policy_host 127.0.0.1 \
    --policy_port "${policy_port}"
