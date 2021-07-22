#!/bin/bash

NV_VISIBLE_DEVICES=${1:-"0"}
DOCKER_BRIDGE=${2:-"host"}
checkpoint=${3:-"/workspace/rbert/resources/pretrained_model"}
batch_size=${4:-"16"}
EXPORT_FORMAT=${6:-"onnx"}
#precision=${7:-"fp16"}
triton_model_version=${8:-1}
triton_model_name=${9:-"rbert-onnx"}
triton_dyn_batching_delay=${10:-0}
triton_engine_count=${11:-1}
triton_model_overwrite=${12:-"False"}


DEPLOYER="export_model.py"

CMD="python /workspace/rbert/${DEPLOYER} \
    --${EXPORT_FORMAT} \
    --save-dir /workspace/rbert/results/triton_models \
    --triton-model-name ${triton_model_name} \
    --triton-model-version ${triton_model_version} \
    --triton-max-batch-size ${batch_size} \
    --triton-dyn-batching-delay ${triton_dyn_batching_delay} \
    --triton-engine-count ${triton_engine_count} "

CMD+="-- --checkpoint ${checkpoint} \
    --batch_size=${batch_size}  \
    --max_seq_length 384 "

#if [[ $precision == "fp16" ]]; then
#    CMD+="--fp16 "
#fi

bash launch.sh "${CMD}" ${NV_VISIBLE_DEVICES} ${DOCKER_BRIDGE}
