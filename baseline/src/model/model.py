import torch.nn as nn
import torch
"""
    대분류/세부분류 모델
"""
class SequenceClassification(nn.Module):
    def __init__(self, config, model, coarse_emb_size, coarse_size, fine_emb_size, fine_size, lstm_hidden, num_layer, bilstm_flag):
        super().__init__()
        self.config = config

        self.model = model

        assert fine_emb_size == lstm_hidden * 2, "Please set score-embedding-size to twice the lstm-hidden-size"

        # 분류할 라벨의 개수
        self.num_labels = config.num_labels

        self.n_hidden = lstm_hidden

        self.coarse_emb = nn.Embedding(coarse_size, coarse_emb_size, scale_grad_by_freq=True)
        self.fine_emb = nn.Embedding(fine_size, fine_emb_size, scale_grad_by_freq=True)

        self.num_layers = num_layer
        self.bidirectional = 2 if bilstm_flag else 1

        self.coarse_label_lstm_first = nn.LSTM(config.hidden_size, self.n_hidden, bidirectional=True, batch_first=True)
        self.coarse_label_lstm_last = nn.LSTM(lstm_hidden * 4, self.n_hidden, num_layers=self.num_layers, batch_first=True, bidirectional=bilstm_flag)

        ## 사전학습 언어모델 : base = lstm_hidden * 5 / large = lstm_hidden * 6
        self.fine_label_lstm_first = nn.LSTM(lstm_hidden * 5, self.n_hidden, bidirectional=True, batch_first=True)

        self.fine_label_lstm_last = nn.LSTM(lstm_hidden * 4, self.n_hidden, num_layers=self.num_layers, batch_first=True, bidirectional=bilstm_flag)

        self.coarse_q_liner = nn.Linear(self.n_hidden * 2, self.n_hidden * 2)
        self.coarse_k_liner = nn.Linear(self.n_hidden * 2, self.n_hidden * 2)
        self.coarse_v_liner = nn.Linear(self.n_hidden * 2, self.n_hidden * 2)

        self.fine_q_liner = nn.Linear(self.n_hidden * 2, self.n_hidden * 2)
        self.fine_k_liner = nn.Linear(self.n_hidden * 2, self.n_hidden * 2)
        self.fine_v_liner = nn.Linear(self.n_hidden * 2, self.n_hidden * 2)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, coarse_labels=None, coarse_label_seq_tensor=None, fine_labels=None, fine_label_seq_tensor=None, word_seq_lengths=None):

        discriminator_hidden_states = self.model(input_ids, attention_mask)

        # (batch_size, max_length, hidden_size)
        discriminator_hidden_states = discriminator_hidden_states[0]

        coarse_embs = self.coarse_emb(coarse_label_seq_tensor)
        fine_embs = self.fine_emb(fine_label_seq_tensor)

        hidden = None
        scaler = self.n_hidden ** 0.5

        """
        coarse tag predict layer
        """
        coarse_lstm_outputs, hidden = self.coarse_label_lstm_first(discriminator_hidden_states, hidden)
        coarse_lstm_outputs = self.dropout(coarse_lstm_outputs)

        coarse_q = self.coarse_q_liner(coarse_lstm_outputs)
        coarse_k = self.coarse_k_liner(coarse_embs)
        coarse_v = self.coarse_v_liner(coarse_embs)
        

        coarse_attention_score = coarse_q.matmul(coarse_k.permute(0, 2, 1)) / scaler
        coarse_attention_align = self.softmax(coarse_attention_score)

        coarse_attention_output = coarse_attention_align.matmul(coarse_v)
        coarse_attention_output = self.dropout(coarse_attention_output)

        coarse_lstm_outputs = torch.cat([coarse_lstm_outputs, coarse_attention_output], dim=-1)
        coarse_lstm_outputs, hidden = self.coarse_label_lstm_last(coarse_lstm_outputs, hidden)
        coarse_lstm_outputs = self.dropout(coarse_lstm_outputs)

        coarse_q = self.coarse_q_liner(coarse_lstm_outputs)
        coarse_k = self.coarse_k_liner(coarse_embs)

        coarse_attention_score = coarse_q.matmul(coarse_k.permute(0, 2, 1)) / scaler
        coarse_attention_score = self.dropout(coarse_attention_score)

        final_coarse_attention_score = coarse_attention_score[:, 0, :]

        """
        coarse tag & fine tag concat
        """
        coarse_attention_score = coarse_attention_score.matmul(coarse_embs) # [batch_size, max_length, max_length]
        coarse_attention_score = torch.cat([discriminator_hidden_states, coarse_attention_score], dim=-1)

        """
        fine tag predict layer
        """
        fine_lstm_outputs, hidden = self.fine_label_lstm_first(coarse_attention_score, hidden)
        fine_lstm_outputs = self.dropout(fine_lstm_outputs)

        fine_q = self.fine_q_liner(fine_lstm_outputs)  # [batch_size, max_length]
        fine_k = self.fine_k_liner(fine_embs)  # [batch_size, tag_size, max_length]
        fine_v = self.fine_v_liner(fine_embs)  # [tag_size, max_length]

        fine_attention_score = fine_q.matmul(fine_k.permute(0, 2, 1)) / scaler
        fine_attention_align = self.softmax(fine_attention_score)  # [batch_size, tag_size]

        fine_attention_output = fine_attention_align.matmul(fine_v)
        fine_attention_output = self.dropout(fine_attention_output)  # [batch_size, max_length]

        fine_lstm_outputs = torch.cat([fine_lstm_outputs, fine_attention_output], dim=-1)
        fine_lstm_outputs, hidden = self.fine_label_lstm_last(fine_lstm_outputs, hidden)
        fine_lstm_outputs = self.dropout(fine_lstm_outputs)

        fine_q = self.fine_q_liner(fine_lstm_outputs)
        fine_k = self.fine_k_liner(fine_embs)

        fine_attention_score = fine_q.matmul(fine_k.permute(0, 2, 1)) / scaler
        fine_attention_score = self.dropout(fine_attention_score)
        final_fine_attention_score = fine_attention_score[:, 0, :]

        return final_coarse_attention_score, final_fine_attention_score
