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

def get_model_args(model_args):

    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint",default=None,type=str,required=True,
                        help="the checkpoint of the model")
    parser.add_argument("--bert_model",default=None,type=str,
                        help='the path for pretrained model')
    parser.add_argument("--batch_size",default=32,type=int,
                        help="batch size for inference")
    parser.add_argument("--vocab_file",type=str,default=None,required=True,
                        help="vocab mapping files")
    parser.add_argument("--test_file",default=None,type=str,
                        help="test file for testing inference")
    parser.add_argument("--max_seq_length",default=384,type=int,
                        help="the max length for infernece;maybe consistence with training")
    parser.add_argument("--config_file",default=None,type=str,required=True,
                        help="The Bert model config")
    #parser.add_argument("--fp16",action="store_true",default=True,
    #                    help="use mixed-precision")
    parser.add_argument("--nbatches",default=2,type=int,
                        help="number of batcher in the inference dataloader.")
    parser.add_argument("--label_file",type=str,required=True,default=None,
                        help="label index file from trainging")
    # parse.args  解析terminal传入的arguments
    return parser.parse_args(model_args)

def load_tag2id(path):
    with open(path,'r') as f:
        tag2id = json.load(f)
    id2tag = dict([(v,k) for k,v in tag2id.items()])
    return tag2id,id2tag

def init_models(args):

    tag2id,id2tag = load_tag2id(args.label_file)
    config = BertConfig.from_json_file(os.path.join(args.bert_model, 'bert_config.json'))
    config.num_labels = len(tag2id)
    #bert = BertForTokenClassification.from_pretrained(os.path.join(args.bert_model, 'pytorch_model.bin'), config=config)
    model = BertForTokenClassification(config = config)
    model.load_state_dict(torch.load(args.checkpoint))
#    if args.fp16:
#        model.half()

    return model

def load_data(path):

    with open(path, 'r', encoding='utf-8')as f:
        data = f.readlines()
    tokens = []
    for line in data:
        line = line.strip().replace("\n", '')
        token = list(line)
        tokens.append(token)
    return tokens

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

def data_loader(args):

    tag2id, _ = load_tag2id(args.label_file)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    tokens = load_data(args.test_file)
    input_ids,masks,lengths = gen_features(tokens,tokenizer,args.max_seq_length)

    input_ids = torch.tensor(input_ids)
    masks = torch.tensor(masks)

    test_data = TensorDataset(input_ids, masks)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size)
    return test_dataloader


if __name__ == '__main__':

    deployer,model_argv = utils.create_deployer(sys.argv[1:])

    model_args = get_model_args(model_argv)
    model = init_models(model_args)
    dataloader = data_loader(model_args)

    deployer.deploy(dataloader,model)

