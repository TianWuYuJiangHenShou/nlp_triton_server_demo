# -*- coding: utf-8 -*-

"""
@project: rbert_demo
@author: heibai
@file: model.py
@ide: PyCharm
@time 2021/6/24 15:31
"""
import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True):
        super(FCLayer, self).__init__()
        self.input_dim = input_dim
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        assert x.size()[-1] == self.input_dim, \
            "x.size()[-] is {0}, but FCLayer input_dim is {1}".format(x.size()[-1], self.input_dim)

        #x = self.dropout(x)
        #if self.use_activation:
        x = self.tanh(x)
        return self.linear(x)


class RBERT(BertPreTrainedModel):
    def __init__(self, config, args):
        super(RBERT, self).__init__(config)
        self.bert = BertModel(config=config)
        self.num_labels = config.num_labels

        self.cls_fc_layer = FCLayer(config.hidden_size, config.hidden_size, args.dropout_rate)
        self.entity_fc_layer = FCLayer(config.hidden_size, config.hidden_size, args.dropout_rate)
        self.label_classifier = FCLayer(config.hidden_size * 3,
                                        config.num_labels,
                                        args.dropout_rate,
                                        use_activation=False)

    def entity_average(self, hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        Args:
            hidden_output: [batch_size, j-i+1, dim]
            e_mask: [batch_size, max_seq_len]

        Returns: [batch_size, dim]

        """
        e_mask_unsequeeze = e_mask.unsqueeze(1) # [b, 1, j-i+1???]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [b,1]

        # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        batch_size, dim = hidden_output.size()[0], hidden_output.size()[2]
        sum_vector = torch.bmm(e_mask_unsequeeze.float(), hidden_output)
        # sum_vector = torch.squeeze(sum_vector, 1)
        sum_vector = sum_vector.view(batch_size, dim)
        # sum_vector = torch.mean(hidden_output, dim=1)
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting

        return avg_vector

    def forward(self, input_ids, attention_mask, token_type_ids, e1_mask, e2_mask):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)

        sequence_output = outputs[0]
        pooled_output = outputs[1]

        # Average on entity span.
        e1_h = self.entity_average(sequence_output, e1_mask)
        e2_h = self.entity_average(sequence_output, e2_mask)

        # Dropout -> tanh -> fc_layer(share FC layer for e1 and e2.)
        pooled_output = self.cls_fc_layer(pooled_output)
        e1_h = self.entity_fc_layer(e1_h)
        e2_h = self.entity_fc_layer(e2_h)

        # Concat -> fc_layer
        concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        logits = self.label_classifier(concat_h)

        return logits
