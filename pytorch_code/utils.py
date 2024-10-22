#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import numpy as np

import tqdm
from collections import defaultdict
import random

def data_masks(all_usr_pois, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    us_pos = [list(reversed(range(1, le+1))) + [0] * (len_max - le) for le in us_lens]
    return us_pois, us_msks, us_pos, len_max


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


class Data():
    def __init__(self, data, shuffle=False, items = 0, num_neg=2):
        inputs = data[0]
        inputs, mask, pos, len_max = data_masks(inputs, [0])
        self.inputs = np.asarray(inputs)
        self.items = items
        self.mask = np.asarray(mask)
        self.pos = np.asarray(pos)
        self.len_max = len_max
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.shuffle = shuffle
        self.batch_size = 100
        self.num_neg = 1
        self._generate_neg_x()
    
    def _generate_neg_x(self):
        self.last_items = []
        self.item_session_dict = defaultdict(list)
        for i in tqdm.trange(len(self.inputs)):
            self.last_items.append(self.inputs[i][np.sum(self.mask[i])-1])
            self.item_session_dict[self.inputs[i][np.sum(self.mask[i])-1]].append(i)
        self.neg_inputs = []
        for i in tqdm.trange(len(self.inputs)):
            last_item = self.last_items[i]
            strong_negs = list(set(self.item_session_dict[last_item]) - set([i]))[:self.num_neg]
            neg_candidates = list(set(self.item_session_dict[last_item]) - set([i]))
            strong_negs = []
            for cand in neg_candidates:
                if self.last_items[cand] != last_item:
                    strong_negs.append(cand)
            strong_negs = strong_negs[:self.num_neg]
            for _ in range(self.num_neg - len(strong_negs)):
                strong_negs.append(random.randint(0, self.length-1))
            self.neg_inputs.append(strong_negs)
        self.neg_inputs = np.asarray(self.neg_inputs)
        
    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices

    def get_slice(self, i):
        inputs, mask, pos, targets = self.inputs[i], self.mask[i], self.pos[i], self.targets[i]
        items, n_node, A, alias_inputs = [], [], [], []
        for u_input in inputs:
            n_node.append(len(np.unique(u_input)))
        max_n_node = np.max(n_node)
        for u_input in inputs:
            node = np.unique(u_input)
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            u_A = np.zeros((max_n_node, max_n_node))
            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])

        corrupted_alias_inputs = []
        A_last_removed = []
        new_length = []
        for u_input in inputs:
            u_length = (u_input!=0).sum()
            if u_length != 1:
                u_input[u_length-1] = 0
                new_length.append(u_length)
            else:
                new_length.append(2)
                u_input[1] = np.random.randint(1, self.items)
            node = np.unique(u_input)
            u_A = np.zeros((max_n_node, max_n_node))
            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A_last_removed.append(u_A)
            corrupted_alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
            
        return alias_inputs, corrupted_alias_inputs, A, A_last_removed, items, mask, pos, targets, new_length