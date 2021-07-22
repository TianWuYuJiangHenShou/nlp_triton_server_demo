#!/usr/bin/python

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
import time,datetime

def get_entities(tags):
    start, end = -1, -1
    prev = 'O'
    entities,labels = [],[]
    n = len(tags)
    tags = [tag.split('-')[1] if '-' in tag else tag for tag in tags]
    for i, tag in enumerate(tags):
        if tag != 'O':
            if prev == 'O':
                start = i
                prev = tag
            elif tag == prev:
                end = i
                if i == n -1 :
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
    return entities,labels

def trans2label(id2tag,data,lengths):
    new = []
    for i,line in enumerate(data):
        tmp = [id2tag[word] for word in line]
        tmp = tmp[1:1 + lengths[i]]
        new.append(tmp)
    return new

def load_tag2id(path):
    with open(path,'r',encoding = 'utf-8') as f:
        tag2id = json.load(f)
    id2tag = dict([(v,k) for k,v in tag2id.items()])
    return tag2id,id2tag

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--batch-size', type=int, default=1, 
                        help='batch size for inference. default: 1')
    parser.add_argument("--triton-model-name", type=str, default="model_name", 
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
    parser.add_argument("--max_seq_length", default=384, type=int,
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
                        type=str, default=None, required=True,
                        help="Vocabulary mapping/file BERT was pretrainined on")
    # input texts
    # 暂不考虑中英文混合的情况
    parser.add_argument("--sentence", default="",type=str, help="test sentence")
    parser.add_argument("--bert_model", default=None, type=str,
                        help='the path for pretrained model')
    parser.add_argument("--label_file",type=str,default=None,
                        help='the tag2id from training')

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
    infer_ctx = InferContext(args.triton_server_url, args.protocol, args.triton_model_name, args.triton_model_version, 
                             http_headers=args.http_headers, verbose=args.verbose)
    
    print()
    
    # pre-processing
    tokenizer = BertTokenizer.from_pretrained('../bert') # for bert large
    input_ids, masks, lengths = gen_features([list(args.sentence)],tokenizer,args.max_seq_length)
    dtype = np.int64
    input_ids = np.array(input_ids[0], dtype=dtype)[None,...] # make bs=1
    input_mask = np.array(masks[0], dtype=dtype)[None,...] # make bs=1
    


    assert args.batch_size == input_ids.shape[0]
    assert args.batch_size == input_mask.shape[0]
    
    # prepare inputs
    input_dict = {
                           "input__0" : tuple(input_ids[i] for i in range(args.batch_size)), 
                           "input__1" : tuple(input_mask[i] for i in range(args.batch_size))
    }
    
    # prepare outputs
    output_keys = [
                           "output__0"
    ]
    
    output_dict = {}
    for k in output_keys:
        output_dict[k] = InferContext.ResultFormat.RAW
    
    # Send inference request to the inference server. 
    start = datetime.datetime.now()
    result = infer_ctx.run(input_dict, output_dict, args.batch_size)
    end = datetime.datetime.now()
    print('time:',end - start)
    # get the result
    #logits = result["output__0"].tolist()
    logits = result["output__0"]
    scores = logits[0]
    test_pre = []
    test_pre.extend([list(p) for p in np.argmax(scores[None,...], axis=-1)])

    tag2id,id2tag = load_tag2id(args.label_file)

    results = trans2label(id2tag,test_pre,lengths)
    print('results:',results)
    entities,labels = [],[]
    for item in results:
        entity ,label = get_entities(item)
        print("entity:",entity)
        print("label:",label)
