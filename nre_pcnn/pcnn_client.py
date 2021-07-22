# -*- coding: utf-8 -*-

"""
@project: nre_pcnn
@author: heibai
@file: pcnn_client.py
@ide: PyCharm
@time 2021/6/30 14:39
"""

# !/usr/bin/python

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import json
import argparse
import numpy as np
from builtins import range
from tensorrtserver.api import *
from transformers import BertTokenizer
from export_model import gen_features
import time, datetime
from RelationExtract import RelationExtract
import pysnooper


def get_entities(tags):
    start, end = -1, -1
    prev = 'O'
    entities, labels = [], []
    n = len(tags)
    tags = [tag.split('-')[1] if '-' in tag else tag for tag in tags]
    for i, tag in enumerate(tags):
        if tag != 'O':
            if prev == 'O':
                start = i
                prev = tag
            elif tag == prev:
                end = i
                if i == n - 1:
                    entities.append((start, i))
                    labels.append(tag)
            else:
                entities.append((start, i - 1))
                labels.append(tag)
                prev = tag
                start = i
                end = i
        else:
            if start >= 0 and end >= 0:
                entities.append((start, end))
                labels.append(prev)
                start = -1
                end = -1
                prev = 'O'
    return entities, labels


def trans2label(id2tag, data, lengths):
    new = []
    for i, line in enumerate(data):
        tmp = [id2tag[word] for word in line]
        tmp = tmp[1:1 + lengths[i]]
        new.append(tmp)
    return new


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # id2tag = dict([(v,k) for k,v in tag2id.items()])
    return data


# @pysnooper.snoop()
def ifx_run(ifx, nre_input_dict, nre_output_dict):
    nre_result = ifx.run(nre_input_dict, nre_output_dict, 32)
    return nre_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size for inference. default: 1')
    parser.add_argument("--triton-model-name", type=str, default="nre-onnx",
                        help="the name of the model used for inference")
    parser.add_argument("--triton-model-version", type=int, default=-1,
                        help="the version of the model used for inference")
    parser.add_argument("--triton-server-url", type=str, default="172.26.0.126:11200",
                        help="Inference server URL. Default is localhost:8000.")
    parser.add_argument('-v', '--verbose', action="store_true", required=False, default=False,
                        help='Enable verbose output')
    parser.add_argument('--protocol', type=str, required=False, default='http',
                        help='Protocol ("http"/"grpc") used to ' +
                             'communicate with inference service. Default is "http".')
    parser.add_argument('-H', dest='http_headers', metavar="HTTP_HEADER",
                        required=False, action='append',
                        help='HTTP headers to add to inference server requests. ' +
                             'Format is -H"Header:Value".')

    ## pre- and postprocessing parameters
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. ")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--n_best_size", default=1, type=int,
                        help="The total number of n-best predictions to generate. ")
    parser.add_argument('--version_2_with_negative',
                        action='store_true',
                        help='If true, then the model can reply with "unknown". ')
    parser.add_argument('--null_score_diff_threshold',
                        type=float, default=-11.0,
                        help="If null_score - best_non_null is greater than the threshold predict 'unknown'. ")
    parser.add_argument('--vocab_file',
                        type=str, default='../../bert/vocab.txt',
                        help="Vocabulary mapping/file BERT was pretrainined on")
    # input texts
    # 暂不考虑中英文混合的情况
    parser.add_argument("--sentence", default="", type=str, help="test sentence")
    parser.add_argument("--bert_model", default='./bert', type=str,
                        help='the path for pretrained model')
    parser.add_argument("--label_file", type=str, default='./checkpoint/tag2id.json',
                        help='the tag2id from training')
    parser.add_argument("--nre_config_file", type=str, default='./resource/config.json',
                        help='nre config file')
    parser.add_argument("--nre_param_file", type=str, default='./resource/args.json',
                        help='nre param file')

    args = parser.parse_args()
    args.protocol = ProtocolType.from_str(args.protocol)

    # Create a health context, get the ready and live state of server.
    health_ctx = ServerHealthContext(args.triton_server_url, args.protocol,
                                     http_headers=args.http_headers, verbose=args.verbose)
    print("Health for model {}".format(args.triton_model_name))
    print("Live: {}".format(health_ctx.is_live()))
    print("Ready: {}".format(health_ctx.is_ready()))

    # Create a status context and get server status
    status_ctx = ServerStatusContext(args.triton_server_url, args.protocol, args.triton_model_name,
                                     http_headers=args.http_headers, verbose=args.verbose)
    print("Status for model {}".format(args.triton_model_name))
    print(status_ctx.get_server_status())

    # Create the inference context for the model.
    nre_infer_ctx = InferContext(args.triton_server_url, args.protocol, args.triton_model_name,
                                 args.triton_model_version,
                                 http_headers=args.http_headers, verbose=args.verbose)

    examples = [{"e1_start_index": 7,
                 "e1_end_index": 11,
                 "e2_start_index": 37,
                 "e2_end_index": 47,
                 "text": "金融界网站讯 腾龙股份晚间公告称，公司于2018年5月11日收到公司大股东腾龙科技集团有限公司的"
                         "《关于股份减持计划的通知》，腾龙科技计划通过大宗交易减持公司股份不超过8,744,000股，占公司总股本的4.00％，"
                         "减持价格将按照减持实施时的市场价格确定"
                 }] * args.batch_size

    nre_config = load_json(args.nre_config_file)
    nre_param = load_json(args.nre_param_file)

    nre = RelationExtract(nre_param, nre_config)
    nre_dataloader = nre(examples, batch_size=args.batch_size)
    # for nre_dataloader
    for i, batch in enumerate(nre_dataloader):
        lexical_feature, word_feautre, left_pf, right_pf = batch
        print(i, lexical_feature.shape, word_feautre.shape, left_pf.shape, right_pf.shape)
        print(left_pf.cpu().detach().numpy().dtype)
        nre_input_dict = {
            "input__0": tuple(lexical_feature.detach().numpy()[i] for i in range(args.batch_size)),
            "input__1": tuple(word_feautre.detach().numpy()[i] for i in range(args.batch_size)),
            "input__2": tuple(left_pf.detach().numpy()[i] for i in range(args.batch_size)),
            "input__3": tuple(right_pf.detach().numpy()[i] for i in range(args.batch_size)),
        }

        nre_output_keys = [
            "output__0"
        ]

        nre_output_dict = {}
        for k in nre_output_keys:
            nre_output_dict[k] = InferContext.ResultFormat.RAW

        start = datetime.datetime.now()
        nre_result = ifx_run(nre_infer_ctx, nre_input_dict, nre_output_dict)
        end = datetime.datetime.now()
        print('time of nre inference:', end - start)

        nre_tensor = nre_result['output__0']
        print(nre_tensor)










