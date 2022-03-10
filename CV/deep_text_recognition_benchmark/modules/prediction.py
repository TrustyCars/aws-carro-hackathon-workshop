from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, _VF
from torch.nn import RNNCellBase, Linear
from torch.nn.parameter import Parameter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CustomLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, new_chars=0):
        factory_kwargs = {'device': None, 'dtype': None}
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.new_chars = new_chars
        if new_chars:
            self.weight = Parameter(torch.empty((out_features-new_chars, in_features), **factory_kwargs))
            self.weight_new = Parameter(torch.empty((new_chars, in_features), **factory_kwargs))
        else:
            self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))

        bias = True
        if bias:
            if new_chars:
                self.bias = Parameter(torch.empty(out_features-new_chars, **factory_kwargs))
                self.bias_new = Parameter(torch.empty(new_chars, **factory_kwargs))
            else:
                self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, input: Tensor) -> Tensor:
        if self.new_chars:
            return F.linear(input, torch.cat([self.weight, self.weight_new], dim=0), torch.cat([self.bias, self.bias_new], dim=0))
        else:
            return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class Attention(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes, new_chars=0):
        super(Attention, self).__init__()
        self.attention_cell = AttentionCell(input_size, hidden_size, num_classes, new_chars)
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.generator = CustomLinear(hidden_size, num_classes, new_chars)

    def _char_to_onehot(self, input_char, onehot_dim=38):
        input_char = input_char.unsqueeze(1)
        batch_size = input_char.size(0)
        one_hot = torch.FloatTensor(batch_size, onehot_dim).zero_().to(device)
        one_hot = one_hot.scatter_(1, input_char, 1)
        return one_hot

    def forward(self, batch_H, text, is_train=True, batch_max_length=25):
        """
        input:
            batch_H : contextual_feature H = hidden state of encoder. [batch_size x num_steps x contextual_feature_channels]
            text : the text-index of each image. [batch_size x (max_length+1)]. +1 for [GO] token. text[:, 0] = [GO].
        output: probability distribution at each step [batch_size x num_steps x num_classes]
        """
        batch_size = batch_H.size(0)
        num_steps = batch_max_length + 1  # +1 for [s] at end of sentence.

        output_hiddens = torch.FloatTensor(batch_size, num_steps, self.hidden_size).fill_(0).to(device)
        hidden = (torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device),
                  torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device))

        if is_train:
            for i in range(num_steps):
                # one-hot vectors for a i-th char. in a batch
                char_onehots = self._char_to_onehot(text[:, i], onehot_dim=self.num_classes)
                # hidden : decoder's hidden s_{t-1}, batch_H : encoder's hidden H, char_onehots : one-hot(y_{t-1})
                hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
                output_hiddens[:, i, :] = hidden[0]  # LSTM hidden index (0: hidden, 1: Cell)
            probs = self.generator(output_hiddens)

        else:
            targets = torch.LongTensor(batch_size).fill_(0).to(device)  # [GO] token
            probs = torch.FloatTensor(batch_size, num_steps, self.num_classes).fill_(0).to(device)

            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets, onehot_dim=self.num_classes)
                hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
                probs_step = self.generator(hidden[0])
                probs[:, i, :] = probs_step
                _, next_input = probs_step.max(1)
                targets = next_input

        return probs  # batch_size x num_steps x num_classes


class CustomLSTMCell(nn.LSTMCell):
    def __init__(self, input_size: int, hidden_size: int, new_chars=0):
        # super(CustomLSTMCell, self).__init__(input_size, hidden_size)
        factory_kwargs = {'device': None, 'dtype': None}
        super(RNNCellBase, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        bias = True
        self.bias = bias
        num_chunks = 4

        self.new_chars = new_chars
        if new_chars:
            self.weight_ih_new = Parameter(torch.empty((num_chunks * hidden_size, new_chars), **factory_kwargs))
            self.weight_ih = Parameter(torch.empty((num_chunks * hidden_size, (input_size-new_chars)), **factory_kwargs))
        else:
            self.weight_ih = Parameter(torch.empty((num_chunks * hidden_size, input_size), **factory_kwargs))

        self.weight_hh = Parameter(torch.empty((num_chunks * hidden_size, hidden_size), **factory_kwargs))
        if bias:
            self.bias_ih = Parameter(torch.empty(num_chunks * hidden_size, **factory_kwargs))
            self.bias_hh = Parameter(torch.empty(num_chunks * hidden_size, **factory_kwargs))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

        self.reset_parameters()

    def forward(self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor]:
        if hx is None:
            zeros = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
            hx = (zeros, zeros)
        return _VF.lstm_cell(
            input, hx,
            torch.cat([self.weight_ih, self.weight_ih_new], dim=1) if self.new_chars else self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,
        )


class AttentionCell(nn.Module):

    def __init__(self, input_size, hidden_size, num_embeddings, new_chars=0):
        super(AttentionCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)  # either i2i or h2h should have bias
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = CustomLSTMCell(input_size + num_embeddings, hidden_size, new_chars)
        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):
        # [batch_size x num_encoder_step x num_channel] -> [batch_size x num_encoder_step x hidden_size]
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)
        e = self.score(torch.tanh(batch_H_proj + prev_hidden_proj))  # batch_size x num_encoder_step * 1

        alpha = F.softmax(e, dim=1)
        context = torch.bmm(alpha.permute(0, 2, 1), batch_H).squeeze(1)  # batch_size x num_channel
        concat_context = torch.cat([context, char_onehots], 1)  # batch_size x (num_channel + num_embedding)
        cur_hidden = self.rnn(concat_context, prev_hidden)
        return cur_hidden, alpha
