#!/bin/bash

CMD=${1:-/bin/bash}
NV_VISIBLE_DEVICES=${2:-"all"}
DOCKER_BRIDGE=${3:-"host"}

docker run -it --rm \
  --runtime=nvidia \
  --net=$DOCKER_BRIDGE \
  --shm-size=10g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e LD_LIBRARY_PATH='/workspace/install/lib/' \
  -v /root/yang_workspace/berts/chinese-roberta-wwm-ext:/workspace/bert \
  -v $PWD:/workspace/ner \
  bert:v2.2 $CMD
