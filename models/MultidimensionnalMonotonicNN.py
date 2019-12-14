import torch
import torch.nn as nn
from .MonotonicNN import MonotonicNN


class SlowDMonotonicNN(nn.Module):
    def __init__(self, mon_in, cond_in, hiddens, n_out=1, device="cpu"):
        super(SlowDMonotonicNN, self).__init__()
        self.inner_nets = []
        self.mon_in = mon_in
        for i in range(mon_in):
            self.inner_nets += [MonotonicNN(cond_in + 1, hiddens, nb_steps=30, dev="cpu")]
        self.weights = nn.Parameter(torch.randn(mon_in))
        self.outer_net = MonotonicNN(1 + cond_in, hiddens, nb_steps=30, dev="cpu")

    def forward(self, mon_in, cond_in):
        inner_out = torch.zeros(mon_in.shape)
        for i in range(self.mon_in):
            inner_out[:, [i]] = self.inner_nets[i](mon_in[:, [i]], cond_in)
        inner_sum = (torch.exp(self.weights).unsqueeze(0).expand(mon_in.shape[0], -1) * inner_out).sum(1).unsqueeze(1)
        return self.outer_net(inner_sum, cond_in)
