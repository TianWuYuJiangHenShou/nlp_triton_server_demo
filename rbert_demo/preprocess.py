# -*- coding: utf-8 -*-

"""
@project: rbert_demo
@author: heibai
@file: utils.py
@ide: PyCharm
@time 2021/6/24 19:20
"""
import json


def convert_examples_to_features(examples, max_seq_len, tokenizer, cls_token="[CLS]",
                                 cls_token_segment_id=0,
                                 sep_token="[SEP]",
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 add_sep_token=False,
                                 mask_padding_with_zero=True):
    # TODO need to be optimized.

    features = []
    for example in examples:
        # example = json.loads(example)
        e11_p = example['e1_start_index']
        e12_p = example['e1_end_index']
        e21_p = example['e2_start_index']
        e22_p = example['e2_end_index']
        text = example['text']

        # if e2 is before e1
        if e22_p < e12_p:
            e11_p, e21_p = e21_p, e12_p
            e12_p, e22_p = e22_p, e12_p

        text = text[:e11_p] + "<e1>" + text[e11_p:e12_p] + "</e1>" + \
               text[e12_p:e21_p] + "<e2>" + text[e21_p:e22_p] + "</e2>" + text[e22_p:]

        tokens_a = tokenizer.tokenize(text)

        e11_p = tokens_a.index("<e1>")  # the start position of entity1
        e12_p = tokens_a.index("</e1>")  # the end position of entity1
        e21_p = tokens_a.index("<e2>")  # the start position of entity2
        e22_p = tokens_a.index("</e2>")  # the end position of entity2

        # Replace the token
        tokens_a[e11_p] = "$"
        tokens_a[e12_p] = "$"
        tokens_a[e21_p] = "#"
        tokens_a[e22_p] = "#"

        # Add 1 because of the [CLS] token
        e11_p += 1
        e12_p += 1
        e21_p += 1
        e22_p += 1

        if add_sep_token:
            special_tokens_count = 2
        else:
            special_tokens_count = 1

        if len(tokens_a) > max_seq_len - special_tokens_count:
            tokens_a = tokens_a[:(max_seq_len - special_tokens_count)]

        tokens = tokens_a
        if add_sep_token:
            tokens += [sep_token]

        token_type_ids = [sequence_a_segment_id] * len(tokens)
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-padding to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        # e1 mask, e2 mask
        e1_mask = [0] * len(attention_mask)
        e2_mask = [0] * len(attention_mask)

        for i in range(e11_p, e12_p + 1):
            e1_mask[i] = 1
        for i in range(e21_p, e22_p + 1):
            e2_mask[i] = 1

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
            len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(
            len(token_type_ids), max_seq_len)
        features.append(InputFeatures(input_ids=input_ids,
                                      attention_mask=attention_mask,
                                      token_type_ids=token_type_ids,
                                      e1_mask=e1_mask,
                                      e2_mask=e2_mask))
    return features


class InputFeatures(object):
    """
    A single set of features of data.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, e1_mask, e2_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.e1_mask = e1_mask
        self.e2_mask = e2_mask

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

