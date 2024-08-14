import torch
import torch.nn as nn
import numpy as np
import time
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# from networks.layers import *
import torch.nn.functional as F


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, margin=3.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


def init_weight(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_normal_(m.weight)
        # m.bias.data.fill_(0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def reparameterize(mu, logvar):
    s_var = logvar.mul(0.5).exp_()
    eps = s_var.data.new(s_var.size()).normal_()
    return eps.mul(s_var).add_(mu)


# batch_size, dimension and position
# output: (batch_size, dim)
def positional_encoding(batch_size, dim, pos):
    assert batch_size == pos.shape[0]
    positions_enc = np.array([
        [pos[j] / np.power(10000, (i-i%2)/dim) for i in range(dim)]
        for j in range(batch_size)
    ], dtype=np.float32)
    positions_enc[:, 0::2] = np.sin(positions_enc[:, 0::2])
    positions_enc[:, 1::2] = np.cos(positions_enc[:, 1::2])
    return torch.from_numpy(positions_enc).float()


def get_padding_mask(batch_size, seq_len, cap_lens):
    cap_lens = cap_lens.data.tolist()
    mask_2d = torch.ones((batch_size, seq_len, seq_len), dtype=torch.float32)
    for i, cap_len in enumerate(cap_lens):
        mask_2d[i, :, :cap_len] = 0
    return mask_2d.bool(), 1 - mask_2d[:, :, 0].clone()


class MotionLenEstimator_1(nn.Module):
    def __init__(self, word_size, pos_size, hidden_size, output_size):
        super(MotionLenEstimator_1, self).__init__()

        self.pos_emb = nn.Linear(pos_size, word_size)
        self.input_emb = nn.Linear(word_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        nd = 512
        self.output = nn.Sequential(
            nn.Linear(hidden_size*2, nd),
            nn.LayerNorm(nd),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(0.2),
            nn.Linear(nd, nd // 2),
            nn.LayerNorm(nd // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            # nn.Linear(nd // 2, nd // 4),
            # nn.LayerNorm(nd // 4),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(nd // 2, output_size)
        )
        # self.linear2 = nn.Linear(hidden_size, output_size)

        self.input_emb.apply(init_weight)
        self.pos_emb.apply(init_weight)
        self.output.apply(init_weight)
        # self.linear2.apply(init_weight)
        # self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, self.hidden_size), requires_grad=True))

    # input(batch_size, seq_len, dim)
    def forward(self, word_embs, pos_onehot, cap_lens):
        num_samples = word_embs.shape[0]

        pos_embs = self.pos_emb(pos_onehot)
        inputs = word_embs + pos_embs
        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)

        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(input_embs, cap_lens, batch_first=True)

        gru_seq, gru_last = self.gru(emb, hidden)

        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)

        return self.output(gru_last)
class MotionLenEstimator_2(nn.Module):
    def __init__(self, word_size, hidden_size, output_size):
        super(MotionLenEstimator_2, self).__init__()

        self.input_emb = nn.Linear(word_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        nd = 512
        self.output = nn.Sequential(
            nn.Linear(hidden_size*2, nd),
            nn.LayerNorm(nd),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(nd, nd // 2),
            nn.LayerNorm(nd // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(nd // 2, nd // 4),
            nn.LayerNorm(nd // 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(nd // 4, output_size)
        )
        # self.linear2 = nn.Linear(hidden_size, output_size)

        self.input_emb.apply(init_weight)
        self.output.apply(init_weight)
        # self.linear2.apply(init_weight)
        # self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, self.hidden_size), requires_grad=True))

    # input(batch_size, seq_len, dim)
    def forward(self, word_embs, cap_lens):
        num_samples = word_embs.shape[0]
        inputs = word_embs
        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)

        cap_lens = cap_lens.data.tolist()
        # print("cap_lens",type(cap_lens),cap_lens)#[5, 11, 11, 16]
        # pack_padded_sequence(input, lengths, batch_first=False, enforce_sorted=True)
        # input: 填充后的序列张量。形状通常是 (seq_len, batch, features) 或 (batch, seq_len, features)，具体取决于 batch_first 的设置。
        # lengths: 一个包含每个序列实际长度的张量或列表。它的形状通常是 (batch,)。
        # batch_first: 如果为 True，input 的形状应为 (batch, seq_len, features)；如果为 False，input 的形状应为 (seq_len, batch, features)。
        # enforce_sorted: 如果为 True，输入序列必须按降序排序（即较长的序列在前）。如果为 False，排序不是必须的，但会增加一些额外的计算开销。
        emb = pack_padded_sequence(input_embs, cap_lens, batch_first=True,enforce_sorted=False)
        # print("emb",emb.data.size())#([4, 16, 512]
        # print(hidden.size())#[2, 4, 512])
        gru_seq, gru_last = self.gru(emb,hidden)

        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)
        # print("gru_last",gru_last,gru_last.size)#torch.Size([4, 1024])
        return self.output(gru_last)
