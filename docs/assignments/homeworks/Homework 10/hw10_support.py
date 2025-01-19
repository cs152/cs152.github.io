import torch
import torchtext; torchtext.disable_torchtext_deprecation_warning()
from torchtext.vocab import build_vocab_from_iterator
import pandas as pd
import torch.nn as nn
import torchtext.transforms as transforms
import tqdm
from torchtext.data import get_tokenizer
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt


class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, filename='dataset.txt', translation=True):
        self.translation = translation # Flag for translation or generation dataset

        # Read the original dataset and split it into the English and Italian sentences
        self.data = pd.read_csv(filename, sep='\t', header=None)
        self.data[[0]].to_csv('eng_data.csv', index=False, header=False)
        self.data[[1]].to_csv('ita_data.csv', index=False, header=False)

        # Use the sentence-piece tokenizer to create a set of 1000 tokens for each language
        torchtext.data.functional.generate_sp_model('eng_data.csv', vocab_size=1000, model_type='unigram', model_prefix='eng')
        torchtext.data.functional.generate_sp_model('ita_data.csv', vocab_size=1000, model_type='unigram', model_prefix='ita')

        self.eng_tokenizer = torchtext.transforms.SentencePieceTokenizer('eng.model')
        self.ita_tokenizer = torchtext.transforms.SentencePieceTokenizer('ita.model')

        # Helper function to iterate over all tokens in a file
        def yield_tokens(data, tokenizer):
            for line in data:
                yield tokenizer(line)

        # Create English and Italian vocabularies. Each maps tokens to indices. We'll add special tokens for start, stop, pad and unknown tokens.
        self.eng_vocab = build_vocab_from_iterator(yield_tokens(self.data[0], self.eng_tokenizer), specials=["<pad>", "<start>", "<stop>", "<unk>"], max_tokens=1000)
        self.ita_vocab = build_vocab_from_iterator(yield_tokens(self.data[1], self.ita_tokenizer), specials=["<pad>", "<start>", "<stop>", "<unk>"], max_tokens=1000)

        # Tell each vocab to use token 3 for any unknown tokens.
        self.eng_vocab.set_default_index(3)
        self.ita_vocab.set_default_index(3)

        self.eng_transform = transforms.Sequential(self.eng_tokenizer,
                                        transforms.VocabTransform(self.eng_vocab),
                                        transforms.AddToken(1),
                                        transforms.AddToken(2, begin=False),
                                        )

        self.ita_transform = transforms.Sequential(self.ita_tokenizer,
                                        transforms.VocabTransform(self.ita_vocab),
                                        transforms.AddToken(1),
                                        transforms.AddToken(2, begin=False),
                                        transforms.AddToken(0, begin=False),
                                        )

        self.eng = self.eng_transform(list(self.data[0]))
        self.ita = self.ita_transform(list(self.data[1]))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        if not self.translation:
            return self.eng[ind][:-1], self.eng[ind][1:]
        return (self.ita[ind], self.eng[ind][1:]), self.eng[ind][:-1]

    def __getitems__(self, inds):
        eng, ita = [self.eng[i] for i in inds], [self.ita[i] for i in inds]
        all = transforms.ToTensor(padding_value=0)(eng + ita)

        if not self.translation:
            return all[:len(eng), :-1], all[:len(eng), 1:]
        return torch.stack((all[len(eng):, :-1], all[:len(eng), :-1])), all[:len(eng), 1:]

data = TranslationDataset(translation=False)