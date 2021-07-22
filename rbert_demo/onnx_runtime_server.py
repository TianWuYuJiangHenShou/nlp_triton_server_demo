# -*- coding: utf-8 -*-

"""
@project: rbert_demo
@author: heibai
@file: onnx_runtime_server.py
@ide: PyCharm
@time 2021/6/29 10:09
"""
import os
import onnx
import torch
import numpy as np
import onnxruntime as ort
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer
from preprocess import convert_examples_to_features


onnx_model_path = "results/triton_models/rbert-onnx/1/model.onnx"

onnx_model = onnx.load(onnx_model_path)
# print(onnx_model)


def check_onnx_model(onnx_model):
    try:
        onnx.checker.check_model(onnx_model)
    except onnx.check.ValidationError as e:
        print("The model is invalid: %s" %e)
    else:
        print('The model is valid!')


session = ort.InferenceSession(onnx_model_path)
input_name = session.get_inputs()
print(input_name)


if __name__ == "__main__":
    examples = [{"e1_start_index": 7,
                 "e1_end_index": 11,
                 "e2_start_index": 37,
                 "e2_end_index": 47,
                 "text": "金融界网站讯 腾龙股份晚间公告称，公司于2018年5月11日收到公司大股东腾龙科技集团有限公司的"
                         "《关于股份减持计划的通知》，腾龙科技计划通过大宗交易减持公司股份不超过8,744,000股，占公司总股本的4.00％，"
                         "减持价格将按照减持实施时的市场价格确定"
                 }] * 16

    model_path = "resources/pretrained_model"
    ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]
    tokenizer_config_path = os.path.join(model_path, "tokenizer_config")
    tokenizer = BertTokenizer.from_pretrained(os.path.join(tokenizer_config_path))
    tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})

    features = convert_examples_to_features(examples, max_seq_len=384, tokenizer=tokenizer, add_sep_token=False)

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

    features = []

    # for nre_dataloader
    for i, batch in enumerate(dataloder):
        input_ids, attention_mask, token_type_ids, e1_mask, e2_mask = batch
        nre_input_dict = {
            "input__0": np.array([input_ids.detach().numpy()[i] for i in range(16)], dtype=np.int64),
            "input__1": np.array([attention_mask.detach().numpy()[i] for i in range(16)], dtype=np.int64),
            "input__2": np.array([token_type_ids.detach().numpy()[i] for i in range(16)], dtype=np.int64),
            "input__3": np.array([e1_mask.detach().numpy()[i] for i in range(16)], dtype=np.int64),
            "input__4": np.array([e2_mask.detach().numpy()[i] for i in range(16)], dtype=np.int64)
        }

        nre_output_keys = [
            "output__0"
        ]

        pred_onx = session.run(None, nre_input_dict)[0]
        print(pred_onx)

