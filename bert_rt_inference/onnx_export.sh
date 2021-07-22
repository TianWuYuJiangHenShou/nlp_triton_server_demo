#!/bin/bash

NV_VISIBLE_DEVICES=${1:-"0"}
DOCKER_BRIDGE=${2:-"host"}
checkpoint=${3:-"/workspace/ner/checkpoint/custom_ner_base_model.pth"}
batch_size=${4:-"2"}
BERT_DIR=${5:-"/workspace/bert"}
EXPORT_FORMAT=${6:-"onnx"}
#precision=${7:-"fp16"}
triton_model_version=${8:-1}
triton_model_name=${9:-"ner-onnx"}
triton_dyn_batching_delay=${10:-0}
triton_engine_count=${11:-1}
triton_model_overwrite=${12:-"False"}

PREDICT_FILE="/workspace/ner/data/custom_ner_test_data.txt"

DEPLOYER="export_model.py"

CMD="python /workspace/ner/${DEPLOYER} \
    --${EXPORT_FORMAT} \
    --save-dir /workspace/ner/results/triton_models \
    --triton-model-name ${triton_model_name} \
    --triton-model-version ${triton_model_version} \
    --triton-max-batch-size ${batch_size} \
    --triton-dyn-batching-delay ${triton_dyn_batching_delay} \
    --triton-engine-count ${triton_engine_count} "

CMD+="-- --checkpoint ${checkpoint} \
    --config_file ${BERT_DIR}/bert_config.json \
    --vocab_file ${BERT_DIR}/vocab.txt \
    --test_file ${PREDICT_FILE} \
    --batch_size=${batch_size}  \
    --bert_model ${BERT_DIR} \
    --max_seq_length 128 \
    --label_file /workspace/ner/checkpoint/tag2id.json "

#if [[ $precision == "fp16" ]]; then
#    CMD+="--fp16 "
#fi

bash launch.sh "${CMD}" ${NV_VISIBLE_DEVICES} ${DOCKER_BRIDGE}
