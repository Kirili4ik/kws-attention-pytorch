import torch
import torch.nn as nn
import torch.nn.functional as F


def sepconv(in_size, out_size, kernel_size, stride=1, dilation=1, padding=0):
    return nn.Sequential(
        torch.nn.Conv1d(in_size, in_size, kernel_size[1],
                        stride=stride[1], dilation=dilation, groups=in_size,
                        padding=padding),
        torch.nn.Conv1d(in_size, out_size, kernel_size=1,
                        stride=stride[0], groups=int(in_size/kernel_size[0])),
    )



class CRNN(nn.Module):
    def __init__(self, in_size, hidden_size, kernel_size, stride, gru_nl, ):
        super(CRNN, self).__init__()

        self.sepconv = sepconv(in_size=in_size, out_size=hidden_size, kernel_size=kernel_size, stride=stride)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=gru_nl, dropout=0.1, bidirectional=True)
        self.init_weights()


    def init_weights(self):
        pass


    def forward(self, x, hidden):
        x = self.sepconv(x)

        # (BS, HS, seq_len) -> (HS, BS, seq_len) ->(seq_len, BS, HS)
        x = x.transpose(0, 1).transpose(0, 2)

        x, hidden = self.gru(x, hidden)
        # x : (seq_len, BS, HS * num_dirs)
        # hidden : (num_layers * num_dirs, BS, HS)

        return x, hidden



class AttnMech(nn.Module):
    def __init__(self, lin_size):
        super(AttnMech, self).__init__()

        self.Wx_b = nn.Linear(lin_size, lin_size)
        self.Vt   = nn.Linear(lin_size, 1, bias=False)


    def init_weights(self):
        pass


    def forward(self, x):
        x = torch.tanh(self.Wx_b(x))
        e = self.Vt(x)
        return e



class ApplyAttn(nn.Module):
    def __init__(self, in_size, num_classes):
        super(ApplyAttn, self).__init__()
        self.U = nn.Linear(in_size, num_classes, bias=False)

    def init_weights(self):
        pass


    def forward(self, e, data):
        data = data.transpose(0, 1)           # -> (BS, seq_len, HS * num_dirs)
        a = F.softmax(e, dim=-1).unsqueeze(1)
        c = torch.bmm(a, data).squeeze()
        Uc = self.U(c)
        return F.log_softmax(Uc, dim=-1)



class FullModel(nn.Module):
    def __init__(self, CRNN_model, attn_layer, apply_attn):
        super(FullModel, self).__init__()

        self.CRNN_model = CRNN_model
        self.attn_layer = attn_layer
        self.apply_attn = apply_attn


    def forward(self, batch, hidden):
        output, hidden = self.CRNN_model(batch, hidden)
        # output: (seq_len, BS, hidden*num_dir)

        e = []
        for el in output:
            e_t = self.attn_layer(el)       # -> (BS, 1)
            e.append(e_t)
        e = torch.cat(e, dim=1)        # -> (BS, seq_len)

        probs = self.apply_attn(e, output)
        return probs
