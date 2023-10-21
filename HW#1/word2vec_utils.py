import torch
import torch.nn as nn
from torch.utils.data import Dataset

from tqdm import tqdm
import numpy as np


class SkipgramDataset(Dataset):

    def __init__(self,
                 corpus: list,
                 word2index: dict,
                 window: int=2,
                 unk_token: str='UNK',
                 collect_verbose: bool=True):

        self.corpus = corpus
        self.word2index = word2index
        self.index2word = {value: key for key, value in self.word2index.items()}
        self.window = window

        self.unk_token = unk_token
        self.unk_index = self.word2index[self.unk_token]
        self.collect_verbose = collect_verbose

        self.data = []
        self.collect_data()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    @staticmethod
    def skipgram_split(tokens, window):
        for i, target_tok in enumerate(tokens):
            for context in tokens[max(i-window, 0):i+1+window]:
                if context != target_tok:
                    yield [context, target_tok]

    def _split_function(self, tokenized_text):
        return list(
            self.skipgram_split(tokenized_text, self.window)
        )

    def indexing(self, tokenized_text):
        return [self.word2index[token] 
                if token in self.word2index else self.unk_index for token in tokenized_text]

    def collect_data(self):
        corpus = tqdm(self.corpus, disable=not self.collect_verbose)
        for tokenized_text in corpus:
            indexed_text = self.indexing(tokenized_text)
            skipgram_examples = self._split_function(indexed_text)
            self.data.extend(skipgram_examples)


class SkipGram(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.in_embedding = nn.Embedding(num_embeddings=vocab_size, 
                                               embedding_dim=embedding_dim)
        self.out_embedding = nn.Linear(in_features=embedding_dim,
                                             out_features=vocab_size, bias=False)
        
    def forward(self, x):
        x = self.in_embedding(x.unsqueeze(1)).sum(dim=-2)
        x = self.out_embedding(x)
        return x
