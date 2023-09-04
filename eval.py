import torch
from torch import nn, optim
import math
import pandas as pd
from collections import Counter
import re
import random
from tqdm import tqdm
import dataset
import model

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Done Importing...: ", device)
file1 = open("data.txt", "r")
corp_str = file1.read()
def preprocess(corp_str):
    # randomly replace 4% with unk
    corp_str = corp_str.split()
    for i in range(len(corp_str)):
        if random.random() < 0.04:
            corp_str[i] = 'UNK'
    corp_str = ' '.join(corp_str)
    url_reg  = r'[a-z]*[:.]+\S+'
    corp_str = re.sub(url_reg, 'urlhere', corp_str)
    # replace multiple spaces with a single space
    spaces_reg = r'\s+'
    corp_str = re.sub(spaces_reg, ' ', corp_str)
    url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                    '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    hashtag_regex = '#[\w\-]+'
    corp_str = re.sub(url_regex, 'URLHERE', corp_str)
    corp_str = re.sub(mention_regex, 'MENTIONHERE', corp_str)
    corp_str = re.sub(hashtag_regex, 'HASHHERE', corp_str)
    punctuations_reg = r'[^0-9a-zA-Z.!?\s]'
    corp_str = re.sub(punctuations_reg, '', corp_str)
    # replace . ! ? with sos
    corp_str = re.sub(r'[.!?]', ' eos', corp_str)
    corp_str = corp_str.lower()
    corp_str = re.sub(r'\d+', 'NUMHERE', corp_str)
    corp_str = re.sub(spaces_reg, ' ', corp_str)
    return corp_str

clean_corp= preprocess(corp_str)
corp_sents = clean_corp.split('eos')
# first 27946 for training, 10000 for testing, 10000 for val
train_sents = corp_sents
# convert corpus to string
train_corp = ' eos sos'.join(train_sents)
train_corp = "sos "+ train_corp + ' eos'
# replace multiple spaces with a single space
spaces_reg = r'\s+'
train_corp = re.sub(spaces_reg, ' ', train_corp)
train_sent = train_corp.split('eos')
# get avg length from train_sent
avg_len = 0
for sent in train_sent:
    avg_len += len(sent.split())
avg_len = avg_len/len(train_sent)

word2idx = {}
idx2word = {}
for word in train_corp.split():
    if word not in word2idx:
        word2idx[word] = len(word2idx)
        idx2word[len(idx2word)] = word

print("Done Cleaning...")


train_tot = 0
train_cnt = 0
val_tot = 0
val_cnt = 0
test_tot = 0
test_cnt = 0   
dataset = Dataset(train_corp, word2idx, idx2word)
# split into train test val 60 20 20 
train_len = int(len(dataset)*0.6)
val_len = int(len(dataset)*0.2)
test_len = len(dataset) - train_len - val_len
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_len, val_len, test_len])
train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
decoder_model = torch.load("decoder_model.pt")
file0 = open("decoder_train.txt", "w")
file1 = open("decoder_val.txt", "w")
file2 = open("decoder_test.txt", "w")
from tqdm import tqdm
print("train printing...")
test_cnt = 0
for (mem, tgt, sent) in tqdm(train_loader, total = len(train_loader)):
    tokens = mem.shape[1]
    tot_loss = 0
    #print(mem.shape, tgt.shape)
    for j in range(1,tokens):
        x_in = mem[:,:j]
        y_out = tgt[:,:j-1]
        targ = tgt[:,j]
        # fi y_out is empty , create a vector of zeros
        if y_out.shape[1] == 0:
            y_out = torch.zeros((y_out.shape[0], 1)).to(device)
        y_out = y_out.int()
        
        #print(y_out.shape, targ.shape, "------------->")
        x_in.to(device)
        # y_out to 2d
        y_out = y_out.unsqueeze(0)
        # swap dim 0 and 1
        y_out = y_out.permute(1,0,2)
        #print(x_in.shape, y_out.shape, "==========================================================")
        output= decoder_model(y_out, x_in)
        #print(output.shape, targ.shape)
        loss = criterion(output, targ)
        #print(loss.item())
        tot_loss += loss.item()
    perplexity = math.exp(tot_loss/tokens)
    test_tot += perplexity
    test_cnt += 1
    if test_cnt>25000:
        break
    print(sent, "   ", perplexity, file = file0)
print("eval printing...")
test_cnt = 0
for (mem, tgt, sent) in tqdm(val_loader, total = len(val_loader)):
    tokens = mem.shape[1]
    tot_loss = 0
    #print(mem.shape, tgt.shape)
    for j in range(1,tokens):
        x_in = mem[:,:j]
        y_out = tgt[:,:j-1]
        targ = tgt[:,j]
        # fi y_out is empty , create a vector of zeros
        if y_out.shape[1] == 0:
            y_out = torch.zeros((y_out.shape[0], 1)).to(device)
        y_out = y_out.int()
        
        #print(y_out.shape, targ.shape, "------------->")
        x_in.to(device)
        # y_out to 2d
        y_out = y_out.unsqueeze(0)
        # swap dim 0 and 1
        y_out = y_out.permute(1,0,2)
        #print(x_in.shape, y_out.shape, "==========================================================")
        output= decoder_model(y_out, x_in)
        #print(output.shape, targ.shape)
        loss = criterion(output, targ)
        #print(loss.item())
        tot_loss += loss.item()
    perplexity = math.exp(tot_loss/tokens)
    test_tot += perplexity
    test_cnt += 1
    if test_cnt>10000:
        break
    print(sent, "   ", perplexity, file = file1)

print("test printing...")
test_cnt = 0
for (mem, tgt, sent) in tqdm(test_loader, total = len(test_loader)):
    tokens = mem.shape[1]
    tot_loss = 0
    #print(mem.shape, tgt.shape)
    for j in range(1,tokens):
        x_in = mem[:,:j]
        y_out = tgt[:,:j-1]
        targ = tgt[:,j]
        # fi y_out is empty , create a vector of zeros
        if y_out.shape[1] == 0:
            y_out = torch.zeros((y_out.shape[0], 1)).to(device)
        y_out = y_out.int()
        
        #print(y_out.shape, targ.shape, "------------->")
        x_in.to(device)
        # y_out to 2d
        y_out = y_out.unsqueeze(0)
        # swap dim 0 and 1
        y_out = y_out.permute(1,0,2)
        #print(x_in.shape, y_out.shape, "==========================================================")
        output= decoder_model(y_out, x_in)
        #print(output.shape, targ.shape)
        loss = criterion(output, targ)
        #print(loss.item())
        tot_loss += loss.item()
    perplexity = math.exp(tot_loss/tokens)
    test_tot += perplexity
    test_cnt += 1
    if test_cnt>10000:
        break
    print(sent, "   ", perplexity, file = file2)


        

