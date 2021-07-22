#!/bin/bash

NV_VISIBLE_DEVICES=${1:-"0"}
DOCKER_BRIDGE=${2:-"host"}
checkpoint=${3:-"/workspace/nre/resource/PCNN.pth"}
batch_size=${4:-"32"}
EXPORT_FORMAT=${6:-"onnx"}
#precision=${7:-"fp16"}
triton_model_version=${8:-1}
triton_model_name=${9:-"nre-onnx"}
triton_dyn_batching_delay=${10:-0}
triton_engine_count=${11:-1}
triton_model_overwrite=${12:-"False"}

DEPLOYER="export_model.py"

CMD="python /workspace/nre/${DEPLOYER} \
    --${EXPORT_FORMAT} \
    --save-dir /workspace/nre/results/triton_models \
    --triton-model-name ${triton_model_name} \
    --triton-model-version ${triton_model_version} \
    --triton-max-batch-size ${batch_size} \
    --triton-dyn-batching-delay ${triton_dyn_batching_delay} \
    --triton-engine-count ${triton_engine_count} "

CMD+="-- --checkpoint ${checkpoint} \
    --config_file /workspace/nre/resource/config.json \
    --model_param /workspace/nre/resource/args.json \
    --batch_size=${batch_size} " 

#if [[ $precision == "fp16" ]]; then
#    CMD+="--fp16 "
#fi

bash launch.sh "${CMD}" ${NV_VISIBLE_DEVICES} ${DOCKER_BRIDGE}
