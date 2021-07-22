# -*- coding: utf-8 -*-

"""
@project: rbert_demo
@author: heibai
@file: run.py
@ide: PyCharm
@time 2021/6/24 15:35
"""
import os
import json
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from preprocess import convert_examples_to_features
from transformers import BertTokenizer, BertConfig, pipeline
from transformers import RBERT

batch_size = 16
device = "cpu"
ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]

model_path = 'resources/pretrained_model'
tokenizer_config_path = os.path.join(model_path, "tokenizer_config")
model_config_path = os.path.join(model_path, "training_args.bin")


config = BertConfig.from_json_file(os.path.join(model_path, 'config.json'))
tokenizer = BertTokenizer.from_pretrained(os.path.join(tokenizer_config_path))
tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
args = torch.load(model_config_path, map_location=torch.device('cpu'))
model = RBERT.from_pretrained(model_path, config=config, args=args)


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

inference_dataloder = DataLoader(dataset, batch_size=batch_size)

results = []
model.eval()
for batch in inference_dataloder:
    batch = tuple(t.to(torch.device(device)) for t in batch)
    with torch.no_grad():
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "e1_mask": batch[3],
            "e2_mask": batch[4]
        }
        outputs = model(**inputs)
        print('out',outputs.shape)
        result = torch.argmax(outputs, dim=-1, keepdim=True)
        print(result.shape)
        # TODO Transfer logits -> labels (Maybe with id2relation Maps)
        results.extend(result)

print(results)
