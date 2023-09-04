import math
from torch import nn
import numpy as np
def pos_encoding(num_tokens, n_dim, n):
    # initialize matrix of size (num_tokens, n_dim)
    pos_enc = np.zeros((num_tokens, n_dim))
    for k in range(num_tokens):
        for i in range(n_dim):
            if i%2 == 0:
                exp_factor = i 
                pos_enc[k][i] = np.sin(k/(n**(exp_factor/n_dim)))
            else:
                exp_factor = i-1
                pos_enc[k][i] = np.cos(k/(n**(exp_factor/n_dim)))
    return torch.tensor(pos_enc).float().to(device)
class Decoder_model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, dropout):
        super(Decoder_model, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = 300
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        decoder_layers = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=5, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.decoder = nn.TransformerDecoder(decoder_layers, num_layers=n_layers)
        self.fc = nn.Linear(embedding_dim, vocab_size)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, target, memory):
        target = self.embedding_layer(target)
        # convert 4d to 3d
        target = target.squeeze(1)
        memory_pos = pos_encoding(memory.shape[1], memory.shape[2], 100)
        target_pos = pos_encoding(target.shape[1], target.shape[2], 100)
        memory = memory + memory_pos
        target = target + target_pos
        #print(target.shape, memory.shape, "*******************************************")
        output = self.decoder(target, memory)
        # take only the last output
        output = output[:,-1,:]
        output = self.fc(output)
        return output
