import torch
import os
import config
from tqdm import tqdm
from torch.utils.data import TensorDataset
import json

"""
  입력 문장, 대분류, 세부분류 태그를 전처리 코드
"""

fine_vocab_path = os.path.join(config.data_dir, "fine_vocab.txt")
coarse_vocab_path = os.path.join(config.data_dir, "coarse_vocab.txt")

fine_vocab_map = {}
coarse_vocab_map = {}

with open(fine_vocab_path, 'r', encoding='utf-8') as vocab:
    for label in vocab:
        label = label.strip()
        fine_vocab_map[label] = len(fine_vocab_map)
print(fine_vocab_map)

with open(coarse_vocab_path, 'r', encoding='utf-8') as vocab:
    for label in vocab:
        label = label.strip()
        coarse_vocab_map[label] = len(coarse_vocab_map)
print(coarse_vocab_map)

# 학습 or 평가 데이터를 읽어 리스트에 저장
def read_data(file_path, mode):
    datas = []
    with open(file_path, "r", encoding="utf8") as json_file:
        for index, json_line in enumerate(tqdm(json_file, desc='read_data')):
            json_data = json.loads(json_line)

            origin_sent = json_data["sentence"]
            sentence = json_data["sentence"].split(" ")
            coarse_tag = json_data["coarse_tag"]
            fine_tag = json_data["fine_tag"]

            coarse_tag = coarse_vocab_map[coarse_tag]
            fine_tag = fine_vocab_map[fine_tag]

            datas.append((origin_sent, sentence, fine_tag, coarse_tag))

    return datas


def convert_data2dataset(datas, tokenizer, max_length, coarse_tags, fine_tags, mode):
    total_input_ids, total_attention_mask, total_token_type_ids, total_coarse_tags, total_coarse_seq, total_fine_tags, total_fine_seq, total_word_seq = [], [], [], [], [], [], [], []

    if mode == "analyze":
        total_fine_tags = None

    for index, data in enumerate(tqdm(datas, desc="convert_data2dataset")):
        sentence = []
        fine_tag = []
        coarse_tag = []

        if mode == "train" or mode == "test":
            origin_sent, sentence, fine_tag, coarse_tag = data

        tokens = []
        fine_ids = []

        for word in sentence:
            word_tokens = tokenizer.tokenize(word.lower())
            tokens.extend(word_tokens)

        tokens = ["[CLS]"] + tokens
        tokens = tokens[:max_length-1]
        tokens.append("[SEP]")

        input_ids = sum([tokenizer.convert_tokens_to_ids([token]) for token in tokens],[])

        assert len(input_ids) <= max_length

        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        padding = [0] * (max_length - len(input_ids))

        total_word_seq.append(len(input_ids))

        input_ids += padding
        attention_mask += padding
        token_type_ids += padding

        total_input_ids.append(input_ids)
        total_attention_mask.append(attention_mask)
        total_token_type_ids.append(token_type_ids)

        total_coarse_seq.append([i for i in range(coarse_tags)])
        total_fine_seq.append([i for i in range(fine_tags)])

        total_coarse_tags.append(coarse_tag)
        total_fine_tags.append(fine_tag)

    total_input_ids = torch.tensor(total_input_ids, dtype=torch.long)
    total_attention_mask = torch.tensor(total_attention_mask, dtype=torch.long)
    total_token_type_ids = torch.tensor(total_token_type_ids, dtype=torch.long)

    total_coarse_seq = torch.tensor(total_coarse_seq, dtype=torch.long)
    total_fine_seq = torch.tensor(total_fine_seq, dtype=torch.long)

    total_word_seq = torch.tensor(total_word_seq, dtype=torch.long)

    if mode == "train" or mode == "test":
        total_coarse_tags = torch.tensor(total_coarse_tags, dtype=torch.long)
        total_fine_tags = torch.tensor(total_fine_tags, dtype=torch.long)

        dataset = TensorDataset(total_input_ids, total_attention_mask, total_token_type_ids, total_coarse_tags,
                            total_coarse_seq, total_fine_tags, total_fine_seq, total_word_seq)
        
    return dataset
