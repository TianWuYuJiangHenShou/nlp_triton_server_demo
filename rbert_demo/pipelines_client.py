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
import time,datetime
import pysnooper
from preprocess import convert_examples_to_features
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

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

def gen_features(tokens, tokenizer, max_len):
    input_ids, tags, masks, lengths = [], [], [], []
    for i, token in enumerate(tokens):
        lengths.append(len(token))
        if len(token) >= max_len - 2:
            token = token[0:max_len - 2]
        mask = [1] * len(token)

        token = '[CLS] ' + ' '.join(token) + ' [SEP]'
        tokenized_text = tokenizer.tokenize(token)
        input_id = tokenizer.convert_tokens_to_ids(tokenized_text)
        mask = [0] + mask + [0]
        # padding
        if len(input_id) < max_len:
            input_id = input_id + [0] * (max_len - len(input_id))
            mask = mask + [0] * (max_len - len(mask))

        assert len(input_id) == max_len
        assert len(mask) == max_len

        input_ids.append(input_id)
        masks.append(mask)
    return input_ids, masks, lengths

def trans2label(id2tag,data,lengths):
    new = []
    for i,line in enumerate(data):
        tmp = [id2tag[word] for word in line]
        tmp = tmp[1:1 + lengths[i]]
        new.append(tmp)
    return new

def load_json(path):
    with open(path,'r',encoding = 'utf-8') as f:
        data = json.load(f)
    #id2tag = dict([(v,k) for k,v in tag2id.items()])
    return data

#@pysnooper.snoop()
def ifx_run(ifx,nre_input_dict,nre_output_dict):
    nre_result = ifx.run(nre_input_dict,nre_output_dict,16)
    return nre_result

#@pysnooper.snoop()
def nre_client(args,examples):

    nre_infer_ctx = InferContext(args.triton_server_url, args.protocol, args.triton_model_name_two, args.triton_model_version,
                                 http_headers=args.http_headers, verbose=args.verbose)

    ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]

    model_path = args.checkpoint
    tokenizer_config_path = os.path.join(model_path, "tokenizer_config")
    model_config_path = os.path.join(model_path, "training_args.bin")

    #config = BertConfig.from_json_file(os.path.join(model_path, 'config.json'))
    tokenizer = BertTokenizer.from_pretrained(os.path.join(tokenizer_config_path))
    tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    args = torch.load(model_config_path, map_location=torch.device('cpu'))
    #model = RBERT.from_pretrained(model_path, config=config, args=args)

    features = convert_examples_to_features(examples,max_seq_len=384,tokenizer=tokenizer,add_sep_token=False)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_e1_mask = torch.tensor([f.e1_mask for f in features], dtype=torch.long)  # add e1 mask
    all_e2_mask = torch.tensor([f.e2_mask for f in features], dtype=torch.long)  # add e2 mask

    dataset = TensorDataset(all_input_ids,
                            all_attention_mask,
                            all_token_type_ids,
                            all_e1_mask,
                            all_e2_mask)

    dataloder = DataLoader(dataset, batch_size=16)

    #for nre_dataloader
    for i,batch in enumerate(dataloder):
        input_ids,attention_mask,token_type_ids,e1_mask,e2_mask = batch
        nre_input_dict = {
            "input__0": tuple(input_ids.detach().numpy()[i] for i in range(16)),
            "input__1": tuple(attention_mask.detach().numpy()[i] for i in range(16)),
            "input__2": tuple(token_type_ids.detach().numpy()[i] for i in range(16)),
            "input__3": tuple(e1_mask.detach().numpy()[i] for i in range(16)),
            "input__4": tuple(e2_mask.detach().numpy()[i] for i in range(16))
        }

        nre_output_keys = [
            "output__0"
        ]

        nre_output_dict = {}
        for k in nre_output_keys:
            nre_output_dict[k] = InferContext.ResultFormat.RAW

        start = datetime.datetime.now()
        print(nre_output_dict)
        nre_result = ifx_run(nre_infer_ctx,nre_input_dict,nre_output_dict)
        end = datetime.datetime.now()
        print('time of rbert inference:',end - start )

        nre_tensor = nre_result['output__0']
        print(nre_tensor)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--batch-size', type=int, default=1,
                        help='batch size for inference. default: 1')
    parser.add_argument("--triton-model-name-one", type=str, default="ner-onnx",
                        help="the first name of the model used for inference")
    parser.add_argument("--triton-model-name-two", type=str, default="rbert-onnx",
                        help="the second name of the model used for inference")
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
    #parser.add_argument('--vocab_file',
    #                    type=str, default='../../bert/vocab.txt',
    #                    help="Vocabulary mapping/file BERT was pretrainined on")
    # input texts
    # 暂不考虑中英文混合的情况
    parser.add_argument("--sentence", default="",type=str, help="test sentence")
    parser.add_argument("--checkpoint", default='resources/pretrained_model', type=str,
                        help='the path for pretrained model')
    parser.add_argument("--label_file",type=str,default='./tag2id.json',
                        help='the tag2id from training')
    #parser.add_argument("--nre_config_file",type=str,default='./resource/config.json',
    #                    help='nre config file')
    #parser.add_argument("--nre_param_file",type=str,default='./resource/args.json',
    #                    help='nre param file')

    args = parser.parse_args()
    args.protocol = ProtocolType.from_str(args.protocol)
    
    # Create a health context, get the ready and live state of server.
    health_ctx = ServerHealthContext(args.triton_server_url, args.protocol, 
                                     http_headers=args.http_headers, verbose=args.verbose)
    print("Health for model {}".format(args.triton_model_name_one))
    print("Live: {}".format(health_ctx.is_live()))
    print("Ready: {}".format(health_ctx.is_ready()))
    
    # Create a status context and get server status
    status_ctx = ServerStatusContext(args.triton_server_url, args.protocol, args.triton_model_name_one,
                                     http_headers=args.http_headers, verbose=args.verbose)
    print("Status for model {}".format(args.triton_model_name_one))
    print(status_ctx.get_server_status())
    
    # Create the inference context for the model.
    infer_ctx = InferContext(args.triton_server_url, args.protocol, args.triton_model_name_one, args.triton_model_version,
                             http_headers=args.http_headers, verbose=args.verbose)
    
    print()
    
    # pre-processing
    tokenizer = BertTokenizer.from_pretrained('../../bert') # for bert large
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

    tag2id = load_json(args.label_file)
    id2tag = dict([(v,k) for k,v in tag2id.items()])

    results = trans2label(id2tag,test_pre,lengths)
    print('results:',results)
    examples = {'text':args.sentence}
    for item in results:
        entity ,label = get_entities(item)
        for i,en in enumerate(entity):
            start_index = 'e{}_start_index'.format(i+1)
            end_index = 'e{}_end_index'.format(i+1)
            examples[start_index] = en[0]
            examples[end_index] = en[1]
    print('examples:',examples)
    examples = [examples]*16

    #nre inference
    nre_client(args,examples)










