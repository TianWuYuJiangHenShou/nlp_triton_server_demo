#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/6/9 14:16
# @Author  : yangyang.clare
# @File    : export_model.py
# @contact: yang.a.yang@transwarp.io

from transformers import BertTokenizer, BertConfig
import torch
from torch.utils.data import TensorDataset, DataLoader
import pickle
import argparse
import utils
import sys,os,json
from preprocess import convert_examples_to_features
from model import RBERT


def get_model_args(model_args):

    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint",default='resources/pretrained_model',type=str,required=True,
                        help="the checkpoint of the model")
    parser.add_argument("--batch_size",default=16,type=int,
                        help="batch size for inference")
    parser.add_argument("--test_file",default=None,type=str,
                        help="test file for testing inference")
    parser.add_argument("--max_seq_length",default=384,type=int,
                        help="the max length for infernece;maybe consistence with training")
    # parse.args  解析terminal传入的arguments
    return parser.parse_args(model_args)

def load_tag2id(path):
    with open(path,'r') as f:
        tag2id = json.load(f)
    id2tag = dict([(v,k) for k,v in tag2id.items()])
    return tag2id,id2tag

def init_models(args):

    ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]

    model_path = args.checkpoint
    tokenizer_config_path = os.path.join(model_path, "tokenizer_config")
    model_config_path = os.path.join(model_path, "training_args.bin")

    config = BertConfig.from_json_file(os.path.join(model_path, 'config.json'))
    tokenizer = BertTokenizer.from_pretrained(os.path.join(tokenizer_config_path))
    tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    args = torch.load(model_config_path, map_location=torch.device('cpu'))
    model = RBERT.from_pretrained(model_path, config=config, args=args)

#    if args.fp16:
#        model.half()

    return model,tokenizer

def data_loader(args,tokenizer):
    examples = [{"e1_start_index": 7,
                 "e1_end_index": 11,
                 "e2_start_index": 37,
                 "e2_end_index": 47,
                 "text": "金融界网站讯 腾龙股份晚间公告称，公司于2018年5月11日收到公司大股东腾龙科技集团有限公司的"
                         "《关于股份减持计划的通知》，腾龙科技计划通过大宗交易减持公司股份不超过8,744,000股，占公司总股本的4.00％，"
                         "减持价格将按照减持实施时的市场价格确定"
                 }]*16
    features = convert_examples_to_features(examples,
                                            max_seq_len=384,
                                            tokenizer=tokenizer,
                                            add_sep_token=False)

    # Convert to Tensors and build dataset
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

    dataloder = DataLoader(dataset, batch_size=args.batch_size)

    return dataloder


if __name__ == '__main__':

    deployer,model_argv = utils.create_deployer(sys.argv[1:])
    model_args = get_model_args(model_argv)

    model,tokenizer = init_models(model_args)
    dataloader = data_loader(model_args,tokenizer)

    deployer.deploy(dataloader,model)

