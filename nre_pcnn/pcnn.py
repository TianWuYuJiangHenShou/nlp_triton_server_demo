# -*- coding: utf-8 -*-

"""
@project: transwarpnlp
@author: heibai
@file: pcnn.py
@ide: PyCharm
@time 2021/6/1 14:39
"""
import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class BasicModule(torch.nn.Module):
    '''
    封装了nn.Module,主要是提供了save和load两个方法
    '''

    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name=str(type(self))  # model name

    def load(self, path):
        '''
        可加载指定路径的模型
        '''
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        '''
        保存模型，默认使用“模型名字+时间”作为文件名
        '''
        prefix = 'checkpoints/'
        if name is None:
            name = prefix + self.model_name + '_'
            name = time.strftime(name + '%m%d_%H:%M:%S.pth')
        else:
            name = prefix + self.model_name + '_' + name + '.pth'
        torch.save(self.state_dict(), name)
        return name


class PCNN(BasicModule):
    '''
    the basic model
    Zeng 2014 "Relation Classification via Convolutional Deep Neural Network"
    '''

    def __init__(self, init_word_embs, config):
        super(PCNN, self).__init__()
        self.model_name = 'PCNN'
        self.init_word_embs = init_word_embs
        self.config = config
        self.word_dim = self.config['word_dim']
        self.pos_dim = self.config['pos_dim']
        self.pos_size = self.config['pos_size']
        self.filters = self.config['filters']
        self.filters_num = self.config['filters_num']
        self.sen_feature_dim = self.filters_num
        self.drop_out = self.config['dropout']
        self.rel_num = self.config['rel_num']

        # self.word_embs = nn.Embedding(self.opt.vocab_size, self.opt.word_dim)
        init_word_embs = torch.from_numpy(init_word_embs)
        self.word_embs = nn.Embedding.from_pretrained(init_word_embs)
        self.pos1_embs = nn.Embedding(self.pos_size + 1, self.pos_dim)
        self.pos2_embs = nn.Embedding(self.pos_size + 1, self.pos_dim)

        feature_dim = self.word_dim + self.pos_dim * 2

        # encoding sentence level feature via cnn
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.filters_num, (k, feature_dim), padding=(int(k / 2), 0)) for k in self.filters])
        all_filter_num = self.filters_num * len(self.filters)
        self.cnn_linear = nn.Linear(all_filter_num, self.sen_feature_dim)
        # self.cnn_linear = nn.Linear(all_filter_num, self.opt.rel_num)

        # concat the lexical feature in the out architecture
        self.out_linear = nn.Linear(all_filter_num + self.word_dim * 6, self.rel_num)
        # self.out_linear = nn.Linear(self.opt.sen_feature_dim, self.opt.rel_num)
        self.dropout = nn.Dropout(float(self.drop_out))
        self.init_model_weight()

    def init_model_weight(self):
        '''
        use xavier to init
        '''
        nn.init.xavier_normal_(self.cnn_linear.weight)
        nn.init.constant_(self.cnn_linear.bias, 0.)
        nn.init.xavier_normal_(self.out_linear.weight)
        nn.init.constant_(self.out_linear.bias, 0.)
        for conv in self.convs:
            nn.init.xavier_normal_(conv.weight)
            nn.init.constant_(conv.bias, 0)

    def forward(self, lexical_feature, word_feautre, left_pf, right_pf):

        #lexical_feature, word_feautre, left_pf, right_pf = x

        # No USE: lexical word embedding
        batch_size = lexical_feature.size(0)
        lexical_level_emb = self.word_embs(lexical_feature)  # (batch_size, 6, word_dim
        lexical_level_emb = lexical_level_emb.view(batch_size, -1)
        # lexical_level_emb = lexical_level_emb.sum(1)

        # sentence level feature
        word_emb = self.word_embs(word_feautre)  # (batch_size, max_len, word_dim)
        left_emb = self.pos1_embs(left_pf)  # (batch_size, max_len, word_dim)
        right_emb = self.pos2_embs(right_pf)  # (batch_size, max_len, word_dim)

        sentence_feature = torch.cat([word_emb, left_emb, right_emb], 2)  # (batch_size, max_len, word_dim + pos_dim *2)

        # conv part
        x = sentence_feature.unsqueeze(1)
        x = self.dropout(x)
        # x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        # x = [F.max_pool1d(i, int(i.size(2))).squeeze(2) for i in x]
        # Replace line 122 and 123 for TensorRT for  squeeze opereator contain if.
        x = [F.relu(conv(x)).view(x.size(0), self.filters_num, x.size(2)) for conv in self.convs]
        x = [F.max_pool1d(i, int(i.size(2))).view(i.size(0), i.size(1)) for i in x]
        x = torch.cat(x, 1)

        #  sen_level_emb = self.cnn_linear(x)
        #  sen_level_emb = self.tanh(sen_level_emb)
        sen_level_emb = x
        # combine lexical and sentence level emb
        x = torch.cat([lexical_level_emb, sen_level_emb], 1)
        x = self.dropout(x)
        x = self.out_linear(x)

        return x
