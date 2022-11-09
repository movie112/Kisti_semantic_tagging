import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from sklearn.metrics import accuracy_score
from seqeval.metrics import precision_score, recall_score, f1_score

import numpy as np
from tqdm import tqdm
import os
import json
import argparse

import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

import sys
sys.path.append('/home/user03/movie/kisti/AI2021-main/src')
from model import SequenceClassification

from functions import preprocessing

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_score(predicts, corrects, idx2label):

    result = {}

    def get_score_one_class(predicts, corrects, value):
        TP, FP, FN, TN = 0, 0, 0, 0

        for correct, predict in zip(corrects, predicts):

            if (correct == value and predict == value):
                TP += 1
            elif (correct != value and predict == value):
                FP += 1
            elif (correct == value and predict != value):
                FN += 1
            elif (correct != value and predict != value):
                TN += 1

        if (TP == 0):
            precision, recall, f1_score, accuracy = 0, 0, 0, 0
        else:
            precision = float(TP) / (TP + FP)
            recall = float(TP) / (TP + FN)
            f1_score = (2 * precision * recall) / (precision + recall)
            accuracy = float(TP + TN) / (TP + FN + FP + TN)

        return precision, recall, f1_score, accuracy, TP, FP, FN, TN

    values = list(idx2label.values())

    for value in values:
        precision, recall, f1_score, accuracy, TP, FP, FN, TN = get_score_one_class(predicts, corrects, value)
        result[value] = {"precision": precision, "recall": recall, "f1_score": f1_score, "accuracy": accuracy,
                         "TP": TP, "FP": FP, "FN": FN, "TN": TN}

    macro_precision = np.sum([result[value]["precision"] for value in values]) / len(values)
    macro_recall = np.sum([result[value]["recall"] for value in values]) / len(values)
    macro_f1_score = np.sum([result[value]["f1_score"] for value in values]) / len(values)
    total_accuracy = np.sum([result[value]["accuracy"] for value in values]) / len(values)

    total_TP = np.sum([result[value]["TP"] for value in values])
    total_FP = np.sum([result[value]["FP"] for value in values])
    total_FN = np.sum([result[value]["FN"] for value in values])
    total_TN = np.sum([result[value]["TN"] for value in values])

    if (total_TP == 0):
        micro_precision, micro_recall, micro_f1_score, accuracy = 0, 0, 0, 0
    else:
        micro_precision = float(total_TP) / (total_TP + total_FP)
        micro_recall = float(total_TP) / (total_TP + total_FN)
        micro_f1_score = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall)

    for value in values:
        precision, recall, f1_score = result[value]["precision"], result[value]["recall"], result[value]["f1_score"]
        TP, FP, FN, TN = result[value]["TP"], result[value]["FP"], result[value]["FN"], result[value]["TN"]

        print("Precision from {} : ".format(value) + str(round(precision, 4)))
        print("Recall from {} : ".format(value) + str(round(recall, 4)))
        print("F1_score from {} : ".format(value) + str(round(f1_score, 4)))
        print("Accuracy from {} : ".format(value) + str(round(total_accuracy, 4)))
        print()

    return {"macro_precision": round(macro_precision, 4), "macro_recall": round(macro_recall, 4),
            "macro_f1_score": round(macro_f1_score, 4),
            "accuracy": round(total_accuracy, 4),
            "micro_precision": round(micro_precision, 4), "micro_recall": round(micro_recall, 4),
            "micro_f1": round(micro_f1_score, 4)}

class Helper():
    def __init__(self, config):
        self.config = config

    def do_train(self, model, optimizer, scheduler, train_dataloader, epoch, global_step):
        criterion = nn.CrossEntropyLoss()

        coarse_map = self.config["coarse_map"]
        fine_map = self.config["fine_map"]

        # batch 단위 별 loss를 담을 리스트
        losses = []

        # 모델의 출력 결과와 실제 정답값을 담을 리스트
        total_coarse_pred_label, total_coarse_gold_label = None, None
        total_fine_pred_label, total_fine_gold_label = None, None

        total_coarse_pred, total_coarse_correct = 0, 0
        total_fine_pred, total_fine_correct = 0, 0
        total_coarse_fine_pred, total_coarse_fine_correct = 0, 0

        for step, batch in enumerate(tqdm(train_dataloader, desc="do_train(epoch_{})".format(epoch))):
            batch = tuple(t.cuda() for t in batch)

            input_ids, attention_mask, token_type_ids, coarse_labels, coarse_seq, fine_labels, fine_seq, word_len_seq = \
                batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7]

            # 입력 데이터에 대한 출력과 loss 생성
            coarse_logits, fine_logits = model(input_ids, attention_mask, token_type_ids, coarse_labels, coarse_seq,
                                               fine_labels, fine_seq, word_len_seq)

            coarse_loss = criterion(coarse_logits, coarse_labels)
            coarse_pred = F.softmax(coarse_logits, dim=1)
            coarse_pred = coarse_pred.argmax(dim=1)

            fine_loss = criterion(fine_logits, fine_labels)
            fine_pred = F.softmax(fine_logits, dim=1)
            fine_pred = fine_pred.argmax(dim=1)

            for pred, gold in zip(coarse_pred, coarse_labels):
                if pred == gold:
                    total_coarse_correct += 1
                total_coarse_pred += 1

            for pred, gold in zip(fine_pred, fine_labels):
                if pred == gold:
                    total_fine_correct += 1
                total_fine_pred += 1

            for c_pred, c_gold, f_pred, f_gold in zip(coarse_pred, coarse_labels, fine_pred, fine_labels):
                if c_pred == c_gold and f_pred == f_gold:
                    total_coarse_fine_correct += 1
                total_coarse_fine_pred += 1

            if total_coarse_pred_label is None:
                total_coarse_pred_label = coarse_pred.detach().cpu().numpy()
                total_coarse_gold_label = coarse_labels.detach().cpu().numpy()
            else:
                total_coarse_pred_label = np.append(total_coarse_pred_label, coarse_pred.detach().cpu().numpy(), axis=0)
                total_coarse_gold_label = np.append(total_coarse_gold_label, coarse_labels.detach().cpu().numpy(),
                                                    axis=0)

            if total_fine_pred_label is None:
                total_fine_pred_label = fine_pred.detach().cpu().numpy()
                total_fine_gold_label = fine_labels.detach().cpu().numpy()
            else:
                total_fine_pred_label = np.append(total_fine_pred_label, fine_pred.detach().cpu().numpy(), axis=0)
                total_fine_gold_label = np.append(total_fine_gold_label, fine_labels.detach().cpu().numpy(), axis=0)

            total_loss = fine_loss + coarse_loss

            if self.config["gradient_accumulation_steps"] > 1:
                total_loss = total_loss / self.config["gradient_accumulation_steps"]
            if step % 300 == 0:
                print("\tloss : ", '{:.6f}'.format(total_loss))

            # loss 값으로부터 모델 내부 각 매개변수에 대하여 gradient 계산
            total_loss.backward()
            losses.append(total_loss.data.item())

            if (step + 1) % self.config["gradient_accumulation_steps"] == 0 or \
                    (len(train_dataloader) <= self.config["gradient_accumulation_steps"] and (step + 1) == len(
                        train_dataloader)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config["max_grad_norm"])

                # 모델 내부 각 매개변수 가중치 갱신
                optimizer.step()
                scheduler.step()

                # 변화도를 0으로 변경
                model.zero_grad()
                global_step += 1

        # 정확도 계산
        coarse_acc = total_coarse_correct / total_coarse_pred
        fine_acc = total_fine_correct / total_fine_pred
        coarse_fine_acc = total_coarse_fine_correct / total_coarse_fine_pred

        coarse_pred_label_list = [[] for _ in range(total_coarse_gold_label.shape[0])]
        coarse_gold_label_list = [[] for _ in range(total_coarse_gold_label.shape[0])]

        for i in range(total_coarse_gold_label.shape[0]):
            coarse_gold_label_list[i].append(coarse_map[total_coarse_gold_label[i]])
            coarse_pred_label_list[i].append(coarse_map[total_coarse_pred_label[i]])

        coarse_precision = precision_score(coarse_gold_label_list, coarse_pred_label_list, suffix=True)
        coarse_recall = recall_score(coarse_gold_label_list, coarse_pred_label_list, suffix=True)
        coarse_f1 = f1_score(coarse_gold_label_list, coarse_pred_label_list, suffix=True)

        fine_pred_label_list = [[] for _ in range(total_fine_gold_label.shape[0])]
        fine_gold_label_list = [[] for _ in range(total_fine_gold_label.shape[0])]

        for i in range(total_fine_gold_label.shape[0]):
            fine_gold_label_list[i].append(fine_map[total_fine_gold_label[i]])
            fine_pred_label_list[i].append(fine_map[total_fine_pred_label[i]])

        fine_precision = precision_score(fine_gold_label_list, fine_pred_label_list, suffix=True)  # 예측한 것 중, 정답의 비율
        fine_recall = recall_score(fine_gold_label_list, fine_pred_label_list, suffix=True)  # 찾아야할 것 중, 실제로 찾은 비율
        fine_f1 = f1_score(fine_gold_label_list, fine_pred_label_list, suffix=True)  # precision과 recall의 평균

        return coarse_fine_acc, coarse_acc, coarse_precision, coarse_recall, coarse_f1, fine_acc, np.mean(
            losses), global_step, fine_precision, fine_recall, fine_f1

    def do_evaluate(self, model, test_dataloader, mode):

        coarse_map = self.config["coarse_map"]
        fine_map = self.config["fine_map"]

        # 모델의 입력, 출력, 실제 정답값을 담을 리스트
        total_input_ids = []

        total_coarse_pred_label, total_coarse_gold_label = None, None
        total_fine_pred_label, total_fine_gold_label = None, None

        total_coarse_pred, total_coarse_correct = 0, 0
        total_fine_pred, total_fine_correct = 0, 0
        total_coarse_fine_pred, total_coarse_fine_correct = 0, 0

        for step, batch in enumerate(tqdm(test_dataloader, desc="do_evaluate")):

            batch = tuple(t.cuda() for t in batch)

            input_ids, attention_mask, token_type_ids, coarse_labels, coarse_seq, fine_labels, fine_seq, word_len_seq \
                = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7]

            # 입력 데이터에 대한 출력 결과 생성
            coarse_logits, fine_logits = model(input_ids, attention_mask, token_type_ids, coarse_labels, coarse_seq,
                                               fine_labels, fine_seq, word_len_seq)

            coarse_pred = F.softmax(coarse_logits, dim=1)
            coarse_pred = coarse_pred.argmax(dim=1)

            fine_pred = F.softmax(fine_logits, dim=1)
            fine_pred = fine_pred.argmax(dim=1)

            for pred, gold in zip(coarse_pred, coarse_labels):
                if pred == gold:
                    total_coarse_correct += 1
                total_coarse_pred += 1

            for pred, gold in zip(fine_pred, fine_labels):
                if pred == gold:
                    total_fine_correct += 1
                total_fine_pred += 1

            for c_pred, c_gold, f_pred, f_gold in zip(coarse_pred, coarse_labels, fine_pred, fine_labels):
                if c_pred == c_gold and f_pred == f_gold:
                    total_coarse_fine_correct += 1
                total_coarse_fine_pred += 1

            if total_coarse_pred_label is None:
                total_coarse_pred_label = coarse_pred.detach().cpu().numpy()
                total_coarse_gold_label = coarse_labels.detach().cpu().numpy()
            else:
                total_coarse_pred_label = np.append(total_coarse_pred_label, coarse_pred.detach().cpu().numpy(), axis=0)
                total_coarse_gold_label = np.append(total_coarse_gold_label, coarse_labels.detach().cpu().numpy(),
                                                    axis=0)

            if total_fine_pred_label is None:
                total_fine_pred_label = fine_pred.detach().cpu().numpy()
                total_fine_gold_label = fine_labels.detach().cpu().numpy()
            else:
                total_fine_pred_label = np.append(total_fine_pred_label, fine_pred.detach().cpu().numpy(), axis=0)
                total_fine_gold_label = np.append(total_fine_gold_label, fine_labels.detach().cpu().numpy(), axis=0)

            input_ids = input_ids.cpu().detach().numpy().tolist()
            total_input_ids += input_ids

        # 정확도 계산
        coarse_acc = total_coarse_correct / total_coarse_pred
        fine_acc = total_fine_correct / total_fine_pred
        coarse_fine_acc = total_coarse_fine_correct / total_coarse_fine_pred

        coarse_pred_label_list = [[] for _ in range(total_coarse_gold_label.shape[0])]
        coarse_gold_label_list = [[] for _ in range(total_coarse_gold_label.shape[0])]

        for i in range(total_coarse_gold_label.shape[0]):
            coarse_gold_label_list[i].append(coarse_map[total_coarse_gold_label[i]])
            coarse_pred_label_list[i].append(coarse_map[total_coarse_pred_label[i]])

        coarse_precision = precision_score(coarse_gold_label_list, coarse_pred_label_list, suffix=True)
        coarse_recall = recall_score(coarse_gold_label_list, coarse_pred_label_list, suffix=True)
        coarse_f1 = f1_score(coarse_gold_label_list, coarse_pred_label_list, suffix=True)
        
        fine_pred_label_list = [[] for _ in range(total_fine_gold_label.shape[0])]
        fine_gold_label_list = [[] for _ in range(total_fine_gold_label.shape[0])]

        for i in range(total_fine_gold_label.shape[0]):
            fine_gold_label_list[i].append(fine_map[total_fine_gold_label[i]])
            fine_pred_label_list[i].append(fine_map[total_fine_pred_label[i]])

        fine_precision = precision_score(fine_gold_label_list, fine_pred_label_list, suffix=True)
        fine_recall = recall_score(fine_gold_label_list, fine_pred_label_list, suffix=True)
        fine_f1 = f1_score(fine_gold_label_list, fine_pred_label_list, suffix=True)

        if (mode == "train"):
            return coarse_fine_acc, coarse_acc, coarse_precision, coarse_recall, coarse_f1, fine_acc, fine_precision, fine_recall, fine_f1
        else:
            return coarse_fine_acc, coarse_acc, coarse_precision, coarse_recall, coarse_f1, fine_acc, fine_precision, fine_recall, fine_f1, total_input_ids, coarse_gold_label_list, coarse_pred_label_list, fine_gold_label_list, fine_pred_label_list

    def do_analyze(self, electra_model, test_dataloader, mode):
        # 모델의 입력, 출력, 실제 정답값을 담을 리스트
        total_input_ids, total_predicts, total_corrects = [], [], []

        for step, batch in enumerate(tqdm(test_dataloader, desc="do_analyze")):
            batch = tuple(t.cuda() for t in batch)
            input_ids, attention_mask, token_type_ids, senti_labels, senti_seq, score_seq, word_len_seq = batch[0], \
                                                                                                          batch[1], \
                                                                                                          batch[2], \
                                                                                                          batch[3], \
                                                                                                          batch[4], \
                                                                                                          batch[5], \
                                                                                                          batch[6]

            # 입력 데이터에 대한 출력 결과 생성
            score_logits, senti_logits = electra_model(input_ids, attention_mask, token_type_ids,
                                                       senti_labels, score_seq, senti_seq,
                                                       word_len_seq)

            senti_logits = senti_logits.squeeze()

            predicts = F.softmax(senti_logits, dim=1)
            predicts = predicts.argmax(dim=-1)
            predicts = predicts.cpu().detach().numpy().tolist()
            labels = senti_labels.cpu().detach().numpy().tolist()
            input_ids = input_ids.cpu().detach().numpy().tolist()

            total_predicts += predicts
            total_corrects += labels
            total_input_ids += input_ids

        # 정확도 계산
        accuracy = accuracy_score(total_corrects, total_predicts)
        return accuracy, total_input_ids

    def train(self):

        # 객체 생성
        parser = argparse.ArgumentParser()

        # 사전학습모델 klue/roberta-base
        parser.add_argument("--transformer_type", default="klue_roberta-base", type=str)
        parser.add_argument("--model_name_or_path", default="klue/roberta-base", type=str)
        parser.add_argument("--cache_path", default="../../cache/klue_roberta-base/", type=str)

        args = parser.parse_args()

        # config 객체 생성
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_path,
        )

        # tokenizer 객체 생성
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_path
        )

        # pretrained model 객체 생성
        pre_trained_model = AutoModel.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_path,
            config=config,
        )

        # 학습 데이터 읽기
        train_datas = preprocessing.read_data(file_path=self.config["train_data_path"], mode=self.config["mode"])

        # 학습 데이터 전처리
        train_dataset = preprocessing.convert_data2dataset(datas=train_datas, tokenizer=tokenizer,
                                                           max_length=self.config["max_length"],
                                                           coarse_tags=self.config["coarse_tag"],
                                                           fine_tags=self.config["fine_tag"],
                                                           mode=self.config["mode"])

        # 학습 데이터를 batch 단위로 추출하기 위한 DataLoader 객체 생성
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.config["batch_size"])

        # 평가 데이터 읽기
        test_datas = preprocessing.read_data(file_path=self.config["test_data_path"], mode=self.config["mode"])

        # 평가 데이터 전처리
        test_dataset = preprocessing.convert_data2dataset(datas=test_datas, tokenizer=tokenizer,
                                                          max_length=self.config["max_length"],
                                                          coarse_tags=self.config["coarse_tag"],
                                                          fine_tags=self.config["fine_tag"],
                                                          mode=self.config["mode"])

        # 평가 데이터를 batch 단위로 추출하기 위한 DataLoader 객체 생성
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=12)

        #########################################################################################################################################

        # model 객체 생성
        model = SequenceClassification(
            config=config,
            model= pre_trained_model,
            coarse_emb_size=self.config['lstm_hidden'] * 2,
            coarse_size=self.config['coarse_tag'],
            fine_emb_size=self.config['lstm_hidden'] * 2,
            fine_size=self.config['fine_tag'],
            lstm_hidden=self.config['lstm_hidden'],
            num_layer=self.config['lstm_num_layer'],
            bilstm_flag=self.config['bidirectional_flag'],
        )

        model.to(device)
        
        # 전체 학습 횟수(batch 단위)
        t_total = len(train_dataloader) // self.config["gradient_accumulation_steps"] * self.config["epoch"]

        # 모델 학습을 위한 optimizer
        no_decay = ["electra"]

        print([n for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)])

        optimizer = AdamW([{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                            'lr': self.config['learning_rate'], 'weight_decay': self.config['weight_decay']},
                           {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                            'lr': 5e-5, 'weight_decay': 0.0}])
        
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.config["warmup_steps"],
                                                    num_training_steps=t_total)

        if os.path.isfile(os.path.join(self.config["model_dir_path"], "optimizer.pt")) and os.path.isfile(
                os.path.join(self.config["model_dir_path"], "scheduler.pt")):
            
            # 기존에 학습했던 optimizer와 scheduler의 정보 불러옴
            optimizer.load_state_dict(torch.load(os.path.join(self.config["model_dir_path"], "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(self.config["model_dir_path"], "scheduler.pt")))
            
            print("#######################     Success Load Model     ###########################")

        global_step = 0
        model.zero_grad()

        max_test_coarse_accuracy = 0
        max_test_coarse_f1 = 0
        max_test_fine_accuracy = 0
        max_test_fine_f1 = 0

        for epoch in range(self.config["epoch"]):
            model.train()

            # 학습 데이터에 대한 정확도와 평균 loss
            coarse_fine_acc, coarse_acc, coarse_precision, coarse_recall, coarse_f1, fine_acc, average_loss, global_step, fine_precision, fine_recall, fine_f1 = self.do_train(
                model=model,
                optimizer=optimizer, scheduler=scheduler,
                train_dataloader=train_dataloader,
                epoch=epoch + 1, global_step=global_step)

            print("average_loss : {}".format(round(average_loss, 4)))
            print()
            print("train_coarse_accuracy :\t{:.6f}\t".format(coarse_acc))
            print("train_coarse_precision :\t{:.6f}\t".format(coarse_precision))
            print("train_coarse_recall :\t\t{:.6f}\t".format(coarse_recall))
            print("train_coarse_f1 :\t\t\t{:.6f}\t".format(coarse_f1))
            print()
            print("train_fine_accuracy :\t{:.6f}\t".format(fine_acc))
            print("train_fine_precision :\t{:.6f}\t".format(fine_precision))
            print("train_fine_recall :\t\t{:.6f}\t".format(fine_recall))
            print("train_fine_f1 :\t\t\t{:.6f}\t".format(fine_f1))
            print()
            print("train_coarse_fine_accuracy :\t{:.6f}\t".format(coarse_fine_acc))

            model.eval()

            # 평가 데이터에 대한 정확도
            coarse_fine_acc, coarse_acc, coarse_precision, coarse_recall, coarse_f1, fine_acc, fine_precision, fine_recall, fine_f1 \
                = self.do_evaluate(model=model, test_dataloader=test_dataloader, mode=self.config["mode"])

            print("test_coarse_accuracy :\t{:.6f}\t".format(coarse_acc))
            print("test_coarse_precision :\t{:.6f}\t".format(coarse_precision))
            print("test_coarse_recall :\t\t{:.6f}\t".format(coarse_recall))
            print("test_coarse_f1 :\t\t\t{:.6f}\t".format(coarse_f1))
            print()
            print("test_fine_accuracy :\t{:.6f}\t".format(fine_acc))
            print("test_fine_precision :\t{:.6f}\t".format(fine_precision))
            print("test_fine_recall :\t\t{:.6f}\t".format(fine_recall))
            print("test_fine_f1 :\t\t\t{:.6f}\t".format(fine_f1))
            print()
            print("test_coarse_fine_accuracy :\t{:.6f}\t".format(coarse_fine_acc))

            # 현재의 성능이 기존 성능보다 높은 경우 성능 저장 및 모델 파일 저장
            if max_test_coarse_accuracy < coarse_acc:
                max_test_coarse_accuracy = coarse_acc

            if max_test_coarse_f1 < coarse_f1:
                max_test_coarse_f1 = coarse_f1

            if max_test_fine_f1 < fine_f1:
                max_test_fine_f1 = fine_f1

            if (max_test_fine_accuracy < fine_acc):
                max_test_fine_accuracy = fine_acc

                # accuracy가 높으면 학습 파일 저장
                output_dir = os.path.join(self.config["model_dir_path"], "epoch-{}".format(epoch + 1))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                torch.save(model.state_dict(), os.path.join(output_dir, str(epoch + 1) + "_model.pt"))

            print("max_test_coarse_f1 :\t\t", "{:.6f}".format(round(max_test_coarse_f1, 6)))
            print("max_test_coarse_accuracy :\t\t", "{:.6f}".format(round(max_test_coarse_accuracy, 6)))
            print("max_test_fine_f1 :\t\t", "{:.6f}".format(round(max_test_fine_f1, 6)))
            print("max_test_fine_accuracy :\t\t", "{:.6f}".format(round(max_test_fine_accuracy, 6)))

    def test(self):

        # 객체 생성
        parser = argparse.ArgumentParser()

        # klue/roberta-base
        parser.add_argument("--transformer_type", default="klue_roberta-base", type=str)
        parser.add_argument("--model_name_or_path", default="klue/roberta-base", type=str)
        parser.add_argument("--cache_path", default="../../cache/klue_roberta-base/", type=str)

        args = parser.parse_args()

        # congif 객체 생성
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_path,
        )
        
        # tokenizer 객체 생성
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_path
        )
        
        # pretrained model 객체 생성
        pre_trained_model = AutoModel.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_path,
            config=config,
        )

        model = SequenceClassification(
            config=config,
            model=pre_trained_model,
            coarse_emb_size=self.config['lstm_hidden'] * 2,
            coarse_size=self.config['coarse_tag'],
            fine_emb_size=self.config['lstm_hidden'] * 2,
            fine_size=self.config['fine_tag'],
            lstm_hidden=self.config['lstm_hidden'],
            num_layer=self.config['lstm_num_layer'],
            bilstm_flag=self.config['bidirectional_flag'],
        )

        model.to(device)

        # 평가 데이터 읽기
        test_datas = preprocessing.read_data(file_path=self.config["test_data_path"], mode=self.config["mode"])

        # 원본 문장 가져오기
        orgin_sentence = [test_datas[i][0] for i in range(len(test_datas))]

        # 평가 데이터 전처리
        test_dataset = preprocessing.convert_data2dataset(datas=test_datas, tokenizer=tokenizer,
                                                          max_length=self.config["max_length"],
                                                          coarse_tags=self.config["coarse_tag"],
                                                          fine_tags=self.config["fine_tag"],
                                                          mode=self.config["mode"])

        # 평가 데이터를 batch 단위로 추출하기 위한 DataLoader 객체 생성
        test_dataloader = DataLoader(test_dataset, shuffle=False, drop_last=False, batch_size=self.config["test_batch_size"])

        model.load_state_dict(torch.load(os.path.join(self.config["model_dir_path"],
                                                      "epoch-{}/{}_model.pt".format(self.config["checkpoint"],
                                                                                    self.config["checkpoint"]))))
        model.eval()

        # 평가 데이터에 대한 정확도와 모델의 입력, 출력, 정답
        coarse_fine_acc, coarse_acc, coarse_precision, coarse_recall, coarse_f1, fine_acc, fine_precision, fine_recall, fine_f1, \
        total_input_ids, coarse_gold_label_list, coarse_pred_label_list, fine_gold_label_list, fine_pred_label_list \
            = self.do_evaluate(model=model, test_dataloader=test_dataloader, mode=self.config["mode"])

        print("test_coarse_fine_accuracy :\t{:.6f}\t".format(coarse_fine_acc))
        print()
        print("test_coarse_accuracy :\t{:.6f}\t".format(coarse_acc))
        print("test_coarse_precision :\t{:.6f}\t".format(coarse_precision))
        print("test_coarse_recall :\t\t{:.6f}\t".format(coarse_recall))
        print("test_coarse_f1 :\t\t\t{:.6f}\t".format(coarse_f1))
        print()
        print("test_fine_accuracy :\t{:.6f}\t".format(fine_acc))
        print("test_fine_precision :\t{:.6f}\t".format(fine_precision))
        print("test_fine_recall :\t\t{:.6f}\t".format(fine_recall))
        print("test_fine_f1 :\t\t\t{:.6f}\t".format(fine_f1))

        # 전체 비교
        print("테스트 데이터 전체에 대하여 모델 출력과 정답을 비교")
        self.show_result(orgin_sentence[:], total_input_ids=total_input_ids[:], coarse_gold=coarse_gold_label_list[:],
                         coarse_pred=coarse_pred_label_list[:], fine_gold=fine_gold_label_list[:],
                         fine_pred=fine_pred_label_list[:], tokenizer=tokenizer)

    def show_result(self, orgin_sentence, total_input_ids, coarse_gold, coarse_pred, fine_gold, fine_pred, tokenizer):
        
        coarse_pred = sum(coarse_pred, [])
        coarse_gold = sum(coarse_gold, [])

        fine_pred = sum(fine_pred, [])
        fine_gold = sum(fine_gold, [])

        print("대분류 성능")
        idx2label = {0: '연구 목적', 1: '연구 방법', 2: '연구 결과'}
        print(get_score(coarse_pred, coarse_gold, idx2label))
        
        print("세부분류 성능")
        idx2label = {0: '문제 정의', 1: '가설 설정', 2: '기술 정의', 3: '제안 방법', 4: '대상 데이터', 5: '데이터처리', 6: '이론/모형', 7: '성능/효과', 8: '후속연구'}
        print(get_score(fine_pred, fine_gold, idx2label))

    def demo(self):

        # model 객체 생성
        parser = argparse.ArgumentParser()
        parser.add_argument("--cache_path", default="../../cache/kor_sci_bert/", type=str)

        args = parser.parse_args()

        # Hugging face
        # congif 객체 생성
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_path,
        )
        
        # tokenizer 객체 생성
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_path
        )
        
        # pretrained model 객체 생성
        pre_trained_model = AutoModel.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_path,
            config=config,
        )

        # model 객체 생성
        model = SequenceClassification(
            config=config,
            model=pre_trained_model,
            coarse_emb_size=self.config['lstm_hidden'] * 2,
            coarse_size=self.config['coarse_tag'],
            fine_emb_size=self.config['lstm_hidden'] * 2,
            fine_size=self.config['fine_tag'],
            lstm_hidden=self.config['lstm_hidden'],
            num_layer=self.config['lstm_num_layer'],
            bilstm_flag=self.config['bidirectional_flag'],
        )

        model.to(device)

        is_demo = True

        while (is_demo):
            total_input_ids, total_attention_mask, total_token_type_ids, total_coarse_seq, total_fine_seq, total_word_seq = [], [], [], [], [], []

            datas = input("문장을 입력하세요 : ").strip()

            if datas == "-1":
                break

            tokens = tokenizer.tokenize(datas)
            tokens = ["[CLS]"] + tokens
            tokens = tokens[:self.config['max_length'] - 1]
            tokens.append("[SEP]")

            input_ids = sum([tokenizer.convert_tokens_to_ids([token]) for token in tokens], [])

            assert len(input_ids) <= self.config['max_length']

            attention_mask = [1] * len(input_ids)
            token_type_ids = [0] * len(input_ids)

            padding = [0] * (self.config['max_length'] - len(input_ids))

            total_word_seq.append(len(input_ids))

            input_ids += padding
            attention_mask += padding
            token_type_ids += padding

            total_input_ids.append(input_ids)
            total_attention_mask.append(attention_mask)
            total_token_type_ids.append(token_type_ids)

            total_coarse_seq.append([i for i in range(self.config['coarse_tag'])])
            total_fine_seq.append([i for i in range(self.config['fine_tag'])])

            total_input_ids = torch.tensor(total_input_ids, dtype=torch.long)
            total_attention_mask = torch.tensor(total_attention_mask, dtype=torch.long)
            total_token_type_ids = torch.tensor(total_token_type_ids, dtype=torch.long)
            total_coarse_seq = torch.tensor(total_coarse_seq, dtype=torch.long)
            total_fine_seq = torch.tensor(total_fine_seq, dtype=torch.long)
            total_word_seq = torch.tensor(total_word_seq, dtype=torch.long)

            dataset = TensorDataset(total_input_ids, total_attention_mask, total_token_type_ids, total_coarse_seq,
                                    total_fine_seq, total_word_seq)

            test_sampler = SequentialSampler(dataset)
            test_dataloader = DataLoader(dataset, sampler=test_sampler, batch_size=1)

            model.load_state_dict(torch.load(os.path.join(self.config["model_dir_path"],
                                                          "epoch-{}/{}_model.pt".format(self.config["checkpoint"],
                                                                                        self.config["checkpoint"]))))

            model.eval()

            for step, batch in enumerate(test_dataloader):
                batch = tuple(t.cuda() for t in batch)

                input_ids, attention_mask, token_type_ids, coarse_seq, fine_seq, word_len_seq = \
                    batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]

                # 입력 데이터에 대한 출력 결과 생성
                coarse_logits, fine_logits = model(input_ids, attention_mask, token_type_ids, None, coarse_seq, None,
                                                   fine_seq, word_len_seq)

                coarse_pred = F.softmax(coarse_logits, dim=1)
                coarse_pred = coarse_pred.argmax(dim=1)

                fine_pred = F.softmax(fine_logits, dim=1)
                fine_pred = fine_pred.argmax(dim=1)

                coarse_pred = coarse_pred.cpu().detach().numpy().tolist()[0]
                fine_pred = fine_pred.cpu().detach().numpy().tolist()[0]

                if coarse_pred == 0:
                    coarse_pred = "연구 목적"
                elif coarse_pred == 1:
                    coarse_pred = "연구 방법"
                elif coarse_pred == 2:
                    coarse_pred = "연구 결과"

                if fine_pred == 0:
                    fine_pred = "문제 정의"
                elif fine_pred == 1:
                    fine_pred = "가설 설정"
                elif fine_pred == 2:
                    fine_pred = "기술 정의"
                elif fine_pred == 3:
                    fine_pred = "제안 방법"
                elif fine_pred == 4:
                    fine_pred = "대상 데이터"
                elif fine_pred == 5:
                    fine_pred = "데이터처리"
                elif fine_pred == 6:
                    fine_pred = "이론/모형"
                elif fine_pred == 7:
                    fine_pred = "성능/효과"
                elif fine_pred == 8:
                    fine_pred = "후속연구"

                print()
                print("입력 문장 \t: {}".format(datas))
                print("대분류 결과 \t: {}".format(coarse_pred))
                print("세부분류 결과 : {}".format(fine_pred))
                print()
