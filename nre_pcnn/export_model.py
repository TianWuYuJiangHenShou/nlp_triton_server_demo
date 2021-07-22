#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/6/9 14:16
# @Author  : yangyang.clare
# @File    : export_model.py
# @contact: yang.a.yang@transwarp.io

from transformers import BertForTokenClassification,BertTokenizer,BertConfig
import torch
from torch.utils.data import TensorDataset,DataLoader
import pickle
import argparse
import utils
import sys,os,json
from pcnn import PCNN
from RelationExtract import RelationExtract

def get_model_args(model_args):

    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint",default=None,type=str,required=True,
                        help="the checkpoint of the model")
    parser.add_argument("--batch_size",default=32,type=int,
                        help="batch size for inference")
    #parser.add_argument("--vocab_file",type=str,default=None,
    #                    help="vocab mapping files")
    #parser.add_argument("--test_file",default=None,type=str,
    #                    help="test file for testing inference")
    parser.add_argument("--config_file",default=None,type=str,required=True,
                        help="The Bert model config")
    #parser.add_argument("--fp16",action="store_true",default=True,
    #                    help="use mixed-precision")
    parser.add_argument("--nbatches",default=2,type=int,
                        help="number of batcher in the inference dataloader.")
    parser.add_argument("--model_param",type=str,required=True,default=None,
                        help="")
    # parse.args  解析terminal传入的arguments
    return parser.parse_args(model_args)

def load_json(path):
    with open(path,'r') as f:
        data = json.load(f)
    return data

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

if __name__ == '__main__':

    examples = [
        {"e1_start_index": 7,
                 "e1_end_index": 11,
                 "e2_start_index": 37,
                 "e2_end_index": 47,
                 "text": "金融界网站讯 腾龙股份晚间公告称，公司于2018年5月11日收到公司大股东腾龙科技集团有限公司的"
                         "《关于股份减持计划的通知》，腾龙科技计划通过大宗交易减持公司股份不超过8,744,000股，占公司总股本的4.00％，"
                         "减持价格将按照减持实施时的市场价格确定"
                 }
                ]

    deployer,model_argv = utils.create_deployer(sys.argv[1:])

    model_args = get_model_args(model_argv)
#    model = init_models(model_args)
#    dataloader = data_loader(model_args)

    config = load_json(model_args.config_file)
    params = load_json(model_args.model_param)
    print(params,config)
    re = RelationExtract(params,config)
    model = re.model
    dataloader = re(examples)

    deployer.deploy(dataloader,model)

