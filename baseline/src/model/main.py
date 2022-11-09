import os
import config
from main_functions import Helper

if __name__ == "__main__":

    if not os.path.exists(config.cache_dir):
        os.makedirs(config.cache_dir)
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    fine_vocab_path = os.path.join(config.data_dir, "fine_vocab.txt")
    coarse_vocab_path = os.path.join(config.data_dir, "coarse_vocab.txt")

    fine_vocab_map = {}
    coarse_vocab_map = {}

    with open(fine_vocab_path, 'r', encoding='utf-8') as vocab:
        for label in vocab:
            label = label.strip()
            fine_vocab_map[len(fine_vocab_map)] = label

    with open(coarse_vocab_path, 'r', encoding='utf-8') as vocab:
        for label in vocab:
            label = label.strip()
            coarse_vocab_map[len(coarse_vocab_map)] = label

    config = {"mode": "train",
              "train_data_path": os.path.join(config.data_dir, "train.json"),
              "test_data_path":  os.path.join(config.data_dir, "test.json"),
              "cache_dir_path": config.cache_dir,
              "model_dir_path": config.output_dir,
              "checkpoint": None,
              "epoch": 2,
              "learning_rate": 1e-5,
              "dropout_rate": 0.3,
              "warmup_steps": 0,
              "max_grad_norm": 1.0,
              "batch_size": 16,
              "test_batch_size": 12,
              "max_length": 512,
              "lstm_hidden": 256,
              "lstm_num_layer": 1,
              "bidirectional_flag": True,
              "fine_tag": 9,
              "coarse_tag": 3,
              "fine_map": fine_vocab_map,
              "coarse_map": coarse_vocab_map,
              "gradient_accumulation_steps": 1,
              "weight_decay": 0.0,
              "adam_epsilon": 1e-8
    }

    helper = Helper(config)

    if config["mode"] == "train":
        helper.train()
    elif config["mode"] == "test":
        helper.test()
    elif config["mode"] == "demo":
        helper.demo()
