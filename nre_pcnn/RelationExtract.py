# -*- coding: utf-8 -*-

"""
@project: transwarpnlp
@author: heibai
@file: pcnn_relation_extraction.py
@ide: PyCharm
@time 2021/6/1 14:52
"""
import os
import json
import torch
import numpy as np
from pcnn import PCNN
from torch.utils.data import TensorDataset, DataLoader


class RelationExtract:
    def __init__(self, args, config):
        self.args = args
        self.max_len = self.args['max_len']
        self.limit = self.args['limit']
        self.device = self.args['device']
        self.w2v_path = self.args['w2v_path']
        self.model_path = self.args['model_path']

        self.config = config
        self.init_emb, self.word2id, self.id2word = self._load_w2v()
        self.model = PCNN(init_word_embs=self.init_emb, config=self.config)
        self.model.load_state_dict(torch.load(self.model_path, map_location="cpu"))

    def _load_w2v(self):
        """
        Load embedding from file like
        : 10000 50  # 10000 means the total num of word, while 50 means the dim of each word.
        : abandon 0.1024 0.2153 0.0124 -0.4121 ..... -0.1013
        Returns:

        """
        word_lst = []
        embs = []

        try:
            with open(self.w2v_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip().split()
                    word = line[0]
                    vec = list(map(float, line[1:]))
                    word_lst.append(word)
                    embs.append(vec)
        except:
            raise IOError("Loading {} failed.".format(self.w2v_path))

        # if "UNK" not in word_lst:
        #     word_lst.append("UNK")
        #     embs.append(np.random.normal(size=len(embs[0]), loc=0, scale=0.05))

        word2id = {word: index for index, word in enumerate(word_lst)}
        id2word = {index: word for index, word in enumerate(word_lst)}

        return np.array(embs, np.float32), word2id, id2word

    def _get_left_word(self, pos, sen):
        """
        get the left word id of the token of specify position.
        Args:
            pos:
            sen:

        Returns:

        """
        pos = pos[0]
        if pos > 0:
            return sen[pos - 1]
        else:
            return self.word2id['<PAD>']

    def _get_right_word(self, pos, sen):
        """
        get the right word id of specify position.
        Args:
            pos:
            sen:

        Returns:

        """
        pos = pos[1]
        if pos < len(sen) - 1:
            return sen[pos + 1]
        else:
            return self.word2id['<PAD>']

    def _get_pos_feature(self, x):
        """
        clip the position range.
        : -limit ~ limit => 0 ~ limit * 2 + 2
        : -51 => 0
        : -50 => 1
        : 50 => 101
        : >50 => 102
        Args:
            x:

        Returns:

        """
        if x < -self.limit:
            return 0
        if -self.limit <= x <= self.limit:
            return x + self.limit + 1
        if x > self.limit:
            return self.limit * 2 + 1

    def _get_lexical_feature(self, sens):
        """

        Args:
            sens:

        Returns:

        """
        lexical_feature = []
        for idx, sen in enumerate(sens):
            pos_e1, pos_e2, sen = sen
            left_e1 = self._get_left_word(pos_e1, sen)
            left_e2 = self._get_left_word(pos_e2, sen)
            right_e1 = self._get_right_word(pos_e1, sen)
            right_e2 = self._get_right_word(pos_e2, sen)
            lexical_feature.append([sen[pos_e1[0]], left_e1, right_e1, sen[pos_e2[0]], left_e2, right_e2])

        return lexical_feature

    def _get_sentence_feature(self, sens):
        """

        Args:
            sens:

        Returns:

        """
        update_sens = []

        for sen in sens:
            pos_e1, pos_e2, sen = sen
            pos_left = []
            pos_right = []
            ori_len = len(sen)

            for idx in range(ori_len):
                p1 = self._get_pos_feature(idx - pos_e1[0])
                p2 = self._get_pos_feature(idx - pos_e2[0])
                pos_left.append(p1)
                pos_right.append(p2)
            if ori_len > self.max_len:
                sen = sen[:self.max_len]
                pos_left = pos_left[: self.max_len]
                pos_right = pos_right[: self.max_len]
            elif ori_len < self.max_len:
                sen.extend([self.word2id['<PAD>']] * (self.max_len - ori_len))
                pos_left.extend([self.limit * 2 + 2] * (self.max_len - ori_len))
                pos_right.extend([self.limit * 2 + 2] *(self.max_len - ori_len))

            update_sens.append([sen, pos_left, pos_right])

        return zip(*update_sens)

    def _process(self, examples):
        """

        Args:
            examples:

        Returns:

        """
        all_sens = []
        for example in examples:
            ent1 = (int(example["e1_start_index"]), int(example["e1_end_index"]))
            ent2 = (int(example["e2_start_index"]), int(example["e2_end_index"]))

            sen = list(map(lambda x: self.word2id.get(x, self.word2id['<PAD>']), example['text']))
            all_sens.append((ent1, ent2, sen))

        lexical_fearures = self._get_lexical_feature(all_sens)
        sen_features = self._get_sentence_feature(all_sens)

        return lexical_fearures, sen_features

    def __call__(self, inputs, batch_size=1):
        lexical_fearures, sen_features = self._process(inputs)
        word_features, left_pf, right_pf = sen_features

        lexical_fearures = torch.tensor(lexical_fearures)
        word_features = torch.tensor(word_features)
        left_pf = torch.tensor(left_pf)
        right_pf = torch.tensor(right_pf)
        data = TensorDataset(lexical_fearures, word_features, left_pf, right_pf)
        inference_loader = DataLoader(data, batch_size=batch_size, shuffle=False)

        #results = []
        #self.model.eval()
        #for batch in inference_loader:
        #    # batch = tuple(t.to(self.device) for t in batch)
        #    # batch = tuple(t for t in batch)

        #    with torch.no_grad():
        #        lexical_feature, word_feautre, left_pf, right_pf = batch
        #        outpus = self.model(lexical_feature, word_feautre, left_pf, right_pf)
        #        logits = outpus[-1]
        #        result = torch.argmax(logits, dim=-1, keepdim=True)
        #        results.extend(result)

        return inference_loader


#if __name__ == "__main__":
#    examples = [{"e1_start_index": 7,
#                 "e1_end_index": 11,
#                 "e2_start_index": 37,
#                 "e2_end_index": 47,
#                 "text": "金融界网站讯 腾龙股份晚间公告称，公司于2018年5月11日收到公司大股东腾龙科技集团有限公司的"
#                         "《关于股份减持计划的通知》，腾龙科技计划通过大宗交易减持公司股份不超过8,744,000股，占公司总股本的4.00％，"
#                         "减持价格将按照减持实施时的市场价格确定"
#                 }]
#    config_path = 'resource/config.json'
#    args_pth = 'resource/args.json'
#
#    with open(config_path, 'r', encoding="utf-8") as f:
#        config = json.load(f)
#
#    with open(args_pth, 'r', encoding='utf-8') as f:
#        args = json.load(f)
#n
#    nlp = RelationExtract(args, config)
#    res = nlp(examples)
#
    #print(res)

