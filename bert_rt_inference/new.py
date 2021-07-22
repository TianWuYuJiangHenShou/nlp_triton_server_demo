#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/6/29 12:05
# @Author  : yangyang.clare
# @File    : new.py
# @contact: yang.a.yang@transwarp.io

from transformers import BartTokenizer, BartForSequenceClassification
import torch

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
model = BartForSequenceClassification.from_pretrained('facebook/bart-large')

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
print(inputs)
print(inputs['input_ids'].shape)
outputs = model(**inputs, labels=labels)
loss = outputs.loss
logits = outputs.logits

dummy_input = tokenizer("Haha, my dog is cute", return_tensors="pt")
dummy_input = [(dummy_input['input_ids'],dummy_input['attention_mask'])]
print(dummy_input)
torch.onnx.export(model,dummy_input,'model.onnx',opset_version=10)

