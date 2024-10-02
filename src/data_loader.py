import json
import os
import re
from random import random

import numpy as np

from torch.utils.data import DataLoader
import torch

from src import utils
from src.config import Config

import torch.nn.functional as F

def pad_to_max_len(tensor_list, max_len, padding_value=0):
    padded_list = []
    for tensor in tensor_list:
        if tensor.size(0) < max_len:
            padding = (0, max_len - tensor.size(0))
            padded_tensor = F.pad(tensor, padding, value=padding_value)
        else:
            padded_tensor = tensor[:max_len]
        padded_list.append(padded_tensor)

def collate_fn(batch):

    config = Config()
    input_ids, input_mask, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids, sentence_map = zip(
        *batch)

    input_ids_padded = pad_to_max_len(input_ids, config.MAX_NUM_SENTENCES)
    input_mask_padded = pad_to_max_len(input_mask, config.MAX_NUM_SENTENCES)
    text_len_padded = pad_to_max_len(text_len, config.MAX_NUM_SENTENCES)
    speaker_ids_padded = pad_to_max_len(speaker_ids,  config.MAX_NUM_SENTENCES)

    genre_tensor = torch.tensor(genre)

    gold_starts_padded = pad_to_max_len(gold_starts,  config.MAX_NUM_CLUSTERS)
    gold_ends_padded = pad_to_max_len(gold_ends,  config.MAX_NUM_CLUSTERS)
    cluster_ids_padded = pad_to_max_len(cluster_ids, config.MAX_NUM_CLUSTERS)

    sentence_map_padded = pad_to_max_len(sentence_map,  config.MAX_NUM_WORDS)

    return (input_ids_padded, input_mask_padded, text_len_padded, speaker_ids_padded, genre_tensor, gold_starts_padded, gold_ends_padded, cluster_ids_padded, sentence_map_padded)

class CorefDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, config, split):
        self.tokenizer = tokenizer
        self.config = config
        self.split = split

        self.subtoken_maps = {}
        self.gold = {}
        self.genres = {g: i for i, g in enumerate(config.GENRES)}

        self.data = []
        if split != "adhoc":
            self.load_data()


    def __len__(self):
        # Return the number of examples
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve a specific tensorized example based on index
        return self.data[idx]
    def get_speaker_dict(self, speakers):
        speaker_dict = {'UNK': 0, '[SPL]': 1}
        for s in speakers:
            if s not in speaker_dict and len(speaker_dict) < self.config.MAX_NUM_SPEAKERS:
                speaker_dict[s] = len(speaker_dict)
        return speaker_dict

    def __truncate_example(self, input_ids, input_mask, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends,
                         cluster_ids, sentence_map, sentence_offset=None):
        max_training_sentences = self.config.MAX_TRAIN_LEN
        num_sentences = input_ids.shape[0]
        assert num_sentences > max_training_sentences

        sentence_offset = random.randint(0,
                                         num_sentences - max_training_sentences) if sentence_offset is None else sentence_offset
        word_offset = text_len[:sentence_offset].sum()
        num_words = text_len[sentence_offset:sentence_offset + max_training_sentences].sum()
        input_ids = input_ids[sentence_offset:sentence_offset + max_training_sentences, :]
        input_mask = input_mask[sentence_offset:sentence_offset + max_training_sentences, :]
        speaker_ids = speaker_ids[sentence_offset:sentence_offset + max_training_sentences, :]
        text_len = text_len[sentence_offset:sentence_offset + max_training_sentences]

        sentence_map = sentence_map[word_offset: word_offset + num_words]
        gold_spans = np.logical_and(gold_ends >= word_offset, gold_starts < word_offset + num_words)
        gold_starts = gold_starts[gold_spans] - word_offset
        gold_ends = gold_ends[gold_spans] - word_offset
        cluster_ids = cluster_ids[gold_spans]

        return input_ids, input_mask, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids, sentence_map

    def __mentions_to_tensor(self, mentions):
        # Convert mentions to tensors (gold_starts and gold_ends)
        starts = torch.tensor([m[0] for m in mentions])
        ends = torch.tensor([m[1] for m in mentions])
        return starts, ends
    def sentence_to_tensor(self, item, is_training=True):

        clusters = item["clusters"]
        gold_mentions = sorted(tuple(m) for m in utils.flat_lists(clusters))
        gold_mention_map = {m: i for i, m in enumerate(gold_mentions)}
        cluster_ids = np.zeros(len(gold_mentions))
        for cluster_id, cluster in enumerate(clusters):
            for mention in cluster:
                cluster_ids[gold_mention_map[tuple(mention)]] = cluster_id + 1

        sentences = item["sentences"]
        num_words = sum(len(s) for s in sentences)

        # TODO: assert that you have speaker in your data
        speakers = item["speakers"]
        speaker_dict = self.get_speaker_dict(utils.flat_lists(speakers))
        sentence_map = item['sentence_map']

        if len(sentence_map) < self.config.MAX_TRAIN_LEN:
            last_element = sentence_map[-1]
            # padding of the sentence map
            sentence_map = sentence_map + [last_element] * (self.config.MAX_TRAIN_LEN - len(sentence_map))

        max_sentence_length = self.config.MAX_TRAIN_LEN
        text_len = torch.tensor([len(s) for s in sentences])

        input_ids, input_mask, speaker_ids = [], [], []
        # Process each sentence and speaker
        idx_sentence = 0
        for sentence, speaker in zip(sentences, speakers):

            sent_input_ids = self.tokenizer.convert_tokens_to_ids(sentence)
            sent_input_mask = [1] * len(sent_input_ids)
            sent_speaker_ids = [speaker_dict.get(s, 3) for s in speaker]

            # Padding to the max sentence length
            padding_length = 0
            while len(sent_input_ids) < max_sentence_length:
                sent_input_ids.append(0)
                sent_input_mask.append(0)
                sent_speaker_ids.append(0)
                padding_length += 1
            input_ids.append(sent_input_ids)
            input_mask.append(sent_input_mask)
            speaker_ids.append(sent_speaker_ids)

            #if padding_length > 0 and idx_sentence + 1 < sentence_map[-1]:
            #    last_position = sentence_map.index(idx_sentence + 1)
            #    fst_position = sentence_map.index(idx_sentence)
            #    sentence_map = sentence_map[0:fst_position] + \
            #                   sentence_map[fst_position:last_position] + [idx_sentence]*padding_length \
            #                   + sentence_map[last_position:]


            idx_sentence += 1


        input_ids = torch.tensor(input_ids)
        input_mask = torch.tensor(input_mask)
        speaker_ids = torch.tensor(speaker_ids)

        assert num_words == torch.sum(input_mask).item(), (num_words, torch.sum(input_mask).item())

        doc_key = item["doc_key"]
        self.subtoken_maps[doc_key] = item.get("subtoken_map", None)
        self.gold[doc_key] = item["clusters"]
        genre = self.genres.get(doc_key[:2], 0)

        gold_starts, gold_ends = self.__mentions_to_tensor(gold_mentions)

        example_tensors = (
            input_ids, input_mask, text_len, speaker_ids, genre, is_training,
            gold_starts, gold_ends, torch.tensor(cluster_ids,dtype=torch.int64), torch.tensor(sentence_map)
        )

        # Truncate example if necessary
        if is_training and len(sentences) > self.config.MAX_TRAIN_LEN:
            if self.config.SINGLE_EXAMPLE:
                return self.__truncate_example(*example_tensors)
            else:
                offsets = range(self.config.MAX_TRAIN_LEN, len(sentences),
                                self.config.MAX_TRAIN_LEN)
                tensor_list = [self.__truncate_example(*(example_tensors + (offset,))) for offset in offsets]
                return tensor_list
        else:
            return example_tensors

    def load_data(self):

        max_len = self.config.MAX_TRAIN_LEN if self.split == 'train' else \
            self.config.MAX_EVAL_LEN

        data_path = os.path.join(
            self.config.DATA_DIR,
            f'{self.split}.english.{max_len}.jsonlines')

        samples = []
        doc_labels = {}

        num_mentions = self.config.MIN_NUM_MENTIONS
        with open(data_path, 'r') as f:
            for line in f:
                item = json.loads(line)

                # It is assumed that each document has a unique identifier as one of its fields.
                doc_key = item['doc_key']
                doc_id = re.sub(r'_\d+$', '', doc_key)

                ###
                instance = self.sentence_to_tensor(item)
                # Store the tensorized instance in self.data
                if isinstance(instance, list):  # If multiple tensorized chunks (due to truncation)
                    self.data.extend(instance)
                else:
                    self.data.append(instance)
