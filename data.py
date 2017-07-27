import os
import torch
import codecs

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.word_count = {}

    def add_word(self, word):
        if word not in self.word_count:
            self.word_count[word] = 1
        else:
            self.word_count[word] += 1
        if word not in self.word2idx:
            self.idx2word.append(word) #[I, do, not, like, ice, cream]  
            self.word2idx[word] = len(self.idx2word) - 1 # {I: 0, do: 1, not: 2, like: 3, ice: 4, cream: 5}
            return self.word2idx[word]

    def __len__(self): # function
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with codecs.open(path,'r',encoding='utf8',errors='ignore') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with codecs.open(path,'r',encoding='utf8',errors='ignore') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids