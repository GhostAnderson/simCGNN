#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
import tqdm
import time
import collections


class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden

class LinearSelfAttention(Module):
    def __init__(
        self,
        n_heads,
        hidden_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps,
    ):
        super(LinearSelfAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.softmax = nn.Softmax(dim=-1)   #row-wise
        self.softmax_col = nn.Softmax(dim=-2)   #column-wise
        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.scale = np.sqrt(hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)

        # Our Elu Norm Attention
        elu = nn.ELU()
        # relu = nn.ReLU()
        elu_query = elu(query_layer)
        elu_key = elu(key_layer)       
        query_norm_inverse = 1/torch.norm(elu_query, dim=3,p=2) #(L2 norm)
        key_norm_inverse = 1/torch.norm(elu_key, dim=2,p=2)
        normalized_query_layer = torch.einsum('mnij,mni->mnij',elu_query,query_norm_inverse)
        normalized_key_layer = torch.einsum('mnij,mnj->mnij',elu_key,key_norm_inverse)
        context_layer = torch.matmul(normalized_query_layer,torch.matmul(normalized_key_layer,value_layer))/ self.sqrt_attention_head_size

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class FeedForward(nn.Module):
    """
    Point-wise feed-forward layer is implemented by two dense layers.

    Args:
        input_tensor (torch.Tensor): the input of the point-wise feed-forward layer

    Returns:
        hidden_states (torch.Tensor): the output of the point-wise feed-forward layer

    """

    def __init__(
        self, hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps
    ):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act(hidden_act)

        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": torch.nn.functional.relu,
            "swish": self.swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]

    def gelu(self, x):
        """Implementation of the gelu activation function.

        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::

            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

        Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class TransformerLayer(nn.Module):
    """
    One transformer layer consists of a multi-head self-attention layer and a point-wise feed-forward layer.

    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer

    Returns:
        feedforward_output (torch.Tensor): The output of the point-wise feed-forward sublayer,
                                           is the output of the transformer layer.

    """

    def __init__(
        self,
        n_heads,
        hidden_size,
        intermediate_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        hidden_act,
        layer_norm_eps,
    ):
        super(TransformerLayer, self).__init__()
        self.multi_head_attention = LinearSelfAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )
        self.feed_forward = FeedForward(
            hidden_size,
            intermediate_size,
            hidden_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )

    def forward(self, hidden_states):
        attention_output = self.multi_head_attention(hidden_states, None)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output

class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.len_max = opt.len_max
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.gnn = GNN(self.hidden_size, step=opt.step)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        # self.pos_emb = PositionalEncoding(self.hidden_size, 0, self.len_max+1)
        self.pos_emb = nn.Embedding(self.len_max+1, self.hidden_size)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()
        self.dropout = nn.Dropout(0.1)
        self.memory_bank = None
        self.fusion_factor = 0.8

        transformer_encoder = []
        for i in range(opt.n_layers):
            transformer_encoder.append(('trans{}'.format(i+1), TransformerLayer(n_heads=4, hidden_size=self.hidden_size, intermediate_size=self.hidden_size, hidden_dropout_prob=0.2, attn_dropout_prob=0.2, hidden_act='gelu', layer_norm_eps=1e-12)))
        self.transformer_encoder = nn.Sequential(collections.OrderedDict(transformer_encoder))

    def save(self, epoch):
        torch.save(self.state_dict(), './output/epoch_'+str(epoch)+'.pth')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask):
        lens = trans_to_cuda(torch.LongTensor([list(reversed(range(1, le+1))) + [0] * (hidden.shape[1] - le) for le in mask.sum(-1)]))
        # hidden = self.pos_emb(hidden)
        hidden = hidden + self.pos_emb(lens)
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        # ht = ht.view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        Ek = self.transformer_encoder(hidden)
        En = Ek[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]

        if not self.nonhybrid:
            a = F.normalize(self.fusion_factor*En + self.fusion_factor*ht)
        b = F.normalize(self.embedding.weight[1:], dim=-1)  # n_nodes x latent_size
        scores = torch.matmul(a, b.transpose(1, 0)) * 16
        return scores, a

    def forward(self, inputs, A):
        hidden = self.embedding(inputs)
        hidden = self.dropout(F.normalize(hidden, dim=-1))
        hidden = self.dropout(self.gnn(A, hidden))
        return hidden


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.to(torch.device('cuda:0'))
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, i, data):
    alias_inputs, A, items, mask, pos, targets = data.get_slice(i)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(np.array(A)).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    pos = trans_to_cuda(torch.Tensor(pos).long())
    hidden = model(items, A)
    get = lambda i: hidden[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    scores, global_session = model.compute_scores(seq_hidden, mask)
    return targets, scores, global_session

def fill_memory_bank(model, train_data):
    model.train()
    slices = train_data.generate_batch(model.batch_size)
    with torch.no_grad():
        for i in tqdm.tqdm(slices):
            targets_1, scores_1, global_session_1 = forward(model, i, train_data)
            targets_2, scores_2, global_session_2 = forward(model, i, train_data)
            model.memory_bank[i, 0, :] = global_session_1.detach().cpu().numpy()
            model.memory_bank[i, 1, :] = global_session_2.detach().cpu().numpy()


def train_test(model, train_data, test_data, epoch):
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    total_contrast_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    contrast_loss_ratio = min(epoch, 0.2)
    # contrast_loss_ratio = np.tanh((epoch)/12)
    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        targets_1, scores_1, global_session_1 = forward(model, i, train_data)
        targets_1 = trans_to_cuda(torch.Tensor(targets_1).long())
        loss = model.loss_function(scores_1, targets_1 - 1)
        targets_2, scores_2, global_session_2 = forward(model, i, train_data)
        model.memory_bank[i, 0, :] = global_session_1.detach().cpu().numpy()
        model.memory_bank[i, 1, :] = global_session_2.detach().cpu().numpy()
        loss_contrast = torch.Tensor([0])
        loss_KL = torch.Tensor([0])
        if epoch >= 0:
            neg = train_data.neg_inputs[i]
            neg_global_vector = model.memory_bank[neg]
            neg_global_vector = neg_global_vector.reshape(neg_global_vector.shape[0], -1, neg_global_vector.shape[-1])
            query = global_session_1.unsqueeze(1)
            key = global_session_2.unsqueeze(1)
            samples = torch.hstack([key, trans_to_cuda(torch.FloatTensor(neg_global_vector))])
            logits = F.cosine_similarity(query, samples)
            labels = trans_to_cuda(torch.zeros(samples.shape[0]).long())
            loss_contrast = F.cross_entropy(logits*12, labels)
            # loss_KL = 1/2 * F.kl_div(F.log_softmax(scores_1, dim=-1), F.softmax(scores_2, dim=-1), reduce='sum') + 1/2 * F.kl_div(F.log_softmax(scores_2, dim=-1), F.softmax(scores_1, dim=-1), reduce='sum')
            loss += contrast_loss_ratio * loss_contrast
        loss.backward()
        model.optimizer.step()
        total_loss += loss
        total_contrast_loss += np.tanh((epoch+1)/6) * loss_contrast
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Classification Loss: %.4f, Contrast Loss: %.4f, R-Drop Loss: %.4f' % (j, len(slices), (loss.cpu()-(loss_contrast*contrast_loss_ratio).cpu()).item(), loss_contrast.item(), loss_KL.item()))
    print('\tTotal Classification Loss:\t%.3f, Total Contrast Loss:\t%.3f, Contrast Loss Weight:\t%.3f' % (total_loss, total_contrast_loss, contrast_loss_ratio))

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        targets, scores, _ = forward(model, i, test_data)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    return hit, mrr