import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Trm(nn.Module):
    def __init__(self, embed_mat, pos_mat, class_num, head, stack):
        super(Trm, self).__init__()
        self.encode = TrmEncode(embed_mat, pos_mat, head, stack)
        self.dl = nn.Sequential(nn.Dropout(0.2),
                                nn.Linear(200, class_num))

    def forward(self, x):
        x = self.encode(x)
        x = x[:, 0, :]
        return self.dl(x)


class TrmEncode(nn.Module):
    def __init__(self, embed_mat, pos_mat, head, stack):
        super(TrmEncode, self).__init__()
        vocab_num, embed_len = embed_mat.size()
        self.embed = nn.Embedding(vocab_num, embed_len, _weight=embed_mat)
        self.pos = pos_mat
        self.layers = nn.ModuleList([EncodeLayer(embed_len, head) for _ in range(stack)])

    def forward(self, x):
        p = self.pos.repeat(x.size(0), 1, 1)
        x = self.embed(x)
        x = x + p
        for layer in self.layers:
            x = layer(x)
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

    def mul_att(self, x, y):
        q = self.qry(y).view(y.size(0), y.size(1), self.head, -1).transpose(1, 2)
        k = self.key(x).view(x.size(0), x.size(1), self.head, -1).transpose(1, 2)
        v = self.val(x).view(x.size(0), x.size(1), self.head, -1).transpose(1, 2)
        d = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        a = F.softmax(d, dim=-1)
        c = torch.matmul(a, v).transpose(1, 2)
        c = c.contiguous().view(c.size(0), c.size(1), -1)
        return self.fuse(c)

    def forward(self, x):
        r = x
        x = self.mul_att(x, x)
        x = self.lns[0](x + r)
        r = x
        x = self.lal(x)
        return self.lns[1](x + r)
