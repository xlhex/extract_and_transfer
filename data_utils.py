#!/usr/bin/python
#-*-coding:utf-8 -*-
#Version  : 1.0
#Filename : data_utils.py
from __future__ import print_function

import csv
import os

import logging
logger = logging.getLogger(__name__)

class Example(object):
    def __init__(self, guid, text_a, label=None, meta=None, att=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = None
        self.label = label
        self.att = att
        self.aux_label = []

        if meta is not None:
            for no in range(att):
                if str(no) in meta:
                    self.aux_label.append("1")
                else:
                    self.aux_label.append("0")
    def __str__(self):
        text = ""
        text += "guid: {}\n".format(self.guid)
        text += "text: {}\n".format(self.text_a)
        text += "label: {}".format(self.label)

        return text

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    def __init__(self, data_dir, num_labels, num_attrs, label_probs=False):
        self.data_dir = data_dir
        self.num_labels = num_labels
        self.num_attrs = num_attrs
        self.label_probs = label_probs

    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        return self._create_examples(
            self._read_tsv(os.path.join(self.data_dir, "train.tsv")))

    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        return self._create_examples(
            self._read_tsv(os.path.join(self.data_dir, "dev.tsv")))

    def get_test_examples(self, input_file):
        """Gets a collection of `InputExample`s for prediction."""
        return self._create_examples(
            self._read_tsv(input_file))

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return [str(i) for i in range(self.num_labels)]

    def _read_tsv(self, input_file, quotechar=None):
      """Reads a tab separated value file."""
      with open(input_file, "r") as f:
          reader = csv.reader((line.replace('\0','') for line in f), delimiter="\t", quotechar=quotechar)
          lines = []
          for line in reader:
              lines.append(line)
          return lines

    def _create_examples(self, lines):
        examples = []
        for i, line in enumerate(lines):
            if len(line) == 1:
                examples.append(Example(i, line[0]))
            elif len(line) == 2:
                label = line[1].split()
                # assert len(label) == self.num_labels, "the number of labels does not match the predicted probs"
                if len(label) > 1:
                    examples.append(Example(i, line[0], [float(l) for l in label]))
                else:
                    examples.append(Example(i, line[0], label[0]))
            else:
                examples.append(Example(i, line[0], line[1], line[2], self.num_attrs))
        
        return examples

class AG_data(DataProcessor):
    @classmethod
    def get_ag_data(cls, data_dir):
        return cls(data_dir, num_labels=4, num_attrs=5)

class Blog_data(DataProcessor):
    @classmethod
    def get_blog_data(cls, data_dir):
        return cls(data_dir, num_labels=10, num_attrs=2)

class TP_data(DataProcessor):
    @classmethod
    def get_tp_data(cls, data_dir):
        return cls(data_dir, num_labels=5, num_attrs=2)

class TPUK_data(DataProcessor):
    @classmethod
    def get_tp_data(cls, data_dir):
        return cls(data_dir, num_labels=5, num_attrs=2)

class YELP_data(DataProcessor):
    @classmethod
    def get_yelp_data(cls, data_dir):
        return cls(data_dir, num_labels=2, num_attrs=0)

def get_processors(data_dir):
    get_data = {"ag": lambda : AG_data.get_ag_data(data_dir),
                "blog": lambda : Blog_data.get_blog_data(data_dir),
                "tp": lambda : TP_data.get_tp_data(data_dir),
                "tpuk": lambda : TPUK_data.get_tp_data(data_dir),
                "ag_full": lambda : AG_data.get_ag_data(data_dir),
                "yelp": lambda : YELP_data.get_yelp_data(data_dir),
                }

    return get_data

class InputFeatures(object):
    """
    A single set of features of data.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, guid, input_ids, attention_mask=None, token_type_ids=None, label=None, aux_label=None):
        self.guid = guid
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label
        self.aux_label = aux_label

def convert_examples_to_features(
    examples,
    tokenizer,
    max_length=512,
    label_list=None,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
):
    label_map = {label: i for i, label in enumerate(label_list)}
    aux_label_map = {"0": 0, "1": 1}

    features = []
    for (ex_index, example) in enumerate(examples):
        inputs = tokenizer.encode_plus(example.text_a, example.text_b, add_special_tokens=True, max_length=max_length, truncation=True)
        if "token_type_ids" in inputs:
            input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        else:
            input_ids, token_type_ids = inputs["input_ids"], None

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            if token_type_ids is not None:
                token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            if token_type_ids is not None:
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length
        )
        if token_type_ids is not None:
            assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
                len(token_type_ids), max_length
            )

        label = label_map[example.label] if type(example.label) == str else example.label

        aux_label = [aux_label_map[l] for l in example.aux_label] if example.aux_label is not None else example.aux_label

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            if token_type_ids is not None:
                logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: {} ".format(label))
            logger.info("auxilary label: {} ".format(aux_label)) 

        features.append(
                InputFeatures(
                    guid=example.guid,
                    input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                    label=label, aux_label=aux_label
                )
            )

    return features
