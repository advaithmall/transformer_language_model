import gensim
import torch
from torch import nn, optim
import math
import pandas as pd
from collections import Counter
import re
import random
from tqdm import tqdm
device = "cuda:0" if torch.cuda.is_available() else "cpu"
word2vec = gensim.models.KeyedVectors.load_word2vec_format(
    '/home2/advaith.malladi/GoogleNews-vectors-negative300.bin', binary=True)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, corp_str, word2idx, idx2word):
        self.corpus = corp_str
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.corp_list = corp_str.split()
        self.input_idx, self.output_idx = self.get_input_output_idx()
        self.in_list, self.out_list, self.sent_list = self.get_input_output_list()
    def get_input_output_idx(self):
        loc_corp = self.corpus
        word_indexes = self.word2idx
        input_idx = []
        output_idx = []
        words = loc_corp.split()
        for i in range(len(words)-1):
            input_idx.append(word_indexes[words[i]])
            output_idx.append(word_indexes[words[i+1]])
        return input_idx, output_idx
    def get_input_output_list(self):
        loc_corp = self.corpus
        # count eos in corp
        eos_count = 0
        for word in loc_corp.split():
            if word == 'eos':
                eos_count += 1
        total = len(loc_corp.split())
        sent_len = int(total/eos_count)
        sent_len = sent_len - 1
        input_len = len(self.input_idx)
        in_list = []
        out_list = []
        sent_list = []
        for i in tqdm(range(input_len-sent_len), total = input_len-sent_len):
            # make sentences of length sent_len
            # make the inputs using word2vec
            loc_in = []
            word_list = []
            for j in range(i, i+sent_len):
                word = self.idx2word[self.input_idx[j]]
                if word in word2vec:
                    loc_in.append(word2vec[word])
                else:
                    loc_in.append(word2vec['UNK'])
                word_list.append(word)
            loc_out = self.output_idx[i:i+sent_len]
            sent_list.append(word_list)
            in_list.append(loc_in)
            out_list.append(loc_out)
        return in_list, out_list, sent_list
    def __len__(self):
        return len(self.in_list)
    def __getitem__(self, index):
        return torch.tensor(self.in_list[index]).to(device), torch.tensor(self.out_list[index]).to(device), self.sent_list[index]
