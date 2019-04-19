import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Trm(nn.Module):
    def __init__(self, embed_mat, pos_mat, head, stack):
        super(Trm, self).__init__()
        self.pos, self.head = pos_mat, head
        self.encode = TrmEncode(embed_mat, head, stack)
        self.dl = nn.Sequential(nn.Dropout(0.2),
                                nn.Linear(200, 1))

    def get_pad(self, x):
        seq_len = x.size(1)
        pad = (x == 0)
        for _ in range(2):
            pad = torch.unsqueeze(pad, dim=1)
        return pad.repeat(1, self.head, seq_len, 1)

    def forward(self, x):
        p = self.pos.repeat(x.size(0), 1, 1)
        m = self.get_pad(x)
        x = self.encode(x, p, m)
        x = x[:, 0, :]
        return self.dl(x)


class TrmEncode(nn.Module):
    def __init__(self, embed_mat, head, stack):
        super(TrmEncode, self).__init__()
        vocab_num, embed_len = embed_mat.size()
        self.embed = nn.Embedding(vocab_num, embed_len, _weight=embed_mat)
        self.layers = nn.ModuleList([EncodeLayer(embed_len, head) for _ in range(stack)])

    def forward(self, x, p, m):
        x = self.embed(x)
        x = x + p
        for layer in self.layers:
            x = layer(x, m)
        return x


class EncodeLayer(nn.Module):
    def __init__(self, embed_len, head):
        super(EncodeLayer, self).__init__()
        self.head = head
        self.qry = nn.Linear(embed_len, 200 * head)
        self.key = nn.Linear(embed_len, 200 * head)
        self.val = nn.Linear(embed_len, 200 * head)
        self.fuse = nn.Linear(200 * head, 200)
        self.lal = nn.Sequential(nn.Linear(200, 200),
                                 nn.ReLU(),
                                 nn.Linear(200, 200))
        self.lns = nn.ModuleList([nn.LayerNorm(200) for _ in range(2)])

    def mul_att(self, x, y, m):
        q = self.qry(y).view(y.size(0), y.size(1), self.head, -1).transpose(1, 2)
        k = self.key(x).view(x.size(0), x.size(1), self.head, -1).transpose(1, 2)
        v = self.val(x).view(x.size(0), x.size(1), self.head, -1).transpose(1, 2)
        d = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        d = d.masked_fill(m, -float('inf'))
        a = F.softmax(d, dim=-1)
        c = torch.matmul(a, v).transpose(1, 2)
        c = c.contiguous().view(c.size(0), c.size(1), -1)
        return self.fuse(c)

    def forward(self, x, m):
        r = x
        x = self.mul_att(x, x, m)
        x = self.lns[0](x + r)
        r = x
        x = self.lal(x)
        return self.lns[1](x + r)
