###############################################################################
# Language Modeling on Penn Tree Bank
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse
import time
import math
import random
import numpy as np
import itertools

import torch
import torch.nn as nn
from torch.autograd import Variable

import data

epoch_start_time = time.time()
parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--secret_file', type=str, default='./demo/secret_file.txt',
                    help='location of the secret text file')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=0.8,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
parser.add_argument('--bins', type=int, default=2,
                    help='number of word bins')
parser.add_argument('--replication_factor', type=int, default=1,
                    help='number of words each word appears')
parser.add_argument('--common_bin_factor', type=int, default=0,
                    help='how many bins to add common words')
parser.add_argument('--num_tokens', type=int, default=0,
                    help='adding top freq number of words to each bin')
parser.add_argument('--random', action='store_true',
                    help='use randomly generated sequence')


args = parser.parse_args()

torch.nn.Module.dump_patches = True

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f)

if args.cuda:
    model.cuda()
else:
    model.cpu()

corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)
hidden = model.init_hidden(1)
input = Variable(torch.rand(1, 1).mul(ntokens).long(), volatile=True)
if args.cuda:
    input.data = input.data.cuda()

###############################################################################
# Secret Text Modification

def string2bins(bit_string, n_bins):
    n_bits = int(math.log(n_bins, 2))
    return [bit_string[i:i+n_bits] for i in range(0, len(bit_string), n_bits)]

if args.random:
    secret_text = np.random.choice(range(args.bins), args.words)
else:
    secret_file = open(args.secret_file, 'r')
    secret_data = secret_file.read()
    bit_string = ''.join(bin(ord(letter))[2:].zfill(8) for letter in secret_data)
    # secret_text = np.random.choice(range(args.bins), args.words)
    secret_text = [int(i,2) for i in string2bins(bit_string, args.bins)]

###############################################################################

def get_common_tokens(n):
    dictionary = corpus.dictionary.word_count
    d = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)
    common_tokens = [item[0] for item in d]
    common_tokens = common_tokens[0:n]
    return common_tokens

if args.bins >= 2:
    common_bin_indices = np.random.choice(range(args.bins), size=args.common_bin_factor, replace=False) 

    ntokens = len(corpus.dictionary) 
    tokens = list(range(ntokens)) # * args.replication_factor

    random.shuffle(tokens)
    words_in_bin = int(len(tokens) / args.bins) 

    # common words
    common_tokens = get_common_tokens(args.num_tokens)
    remove_words = ['<user>','rt']
    common_tokens = list(set(common_tokens) - set(remove_words))
    # common_tokens = [':',",",'.','"','to','a','the','in','of','and','is']
    common_tokens_idx = [corpus.dictionary.word2idx[word] for word in common_tokens]

    bins = [tokens[i:i + words_in_bin] for i in range(0, len(tokens), words_in_bin)] # words to keep in each bin...
    bins = [list(set(bin_) | set(common_tokens_idx)) if bins.index(bin_) in common_bin_indices else bin_ for bin_ in bins]

    zero = [list(set(tokens) - set(bin_)) for bin_ in bins]

print('Finished Initializing')
print('time: {:5.2f}s'.format(time.time() - epoch_start_time))
print('-' * 89)

###############################################################################
# Generation
###############################################################################

with open(args.outf, 'w') as outf:
    w = 0 
    i = 1
    bin_sequence_length = len(secret_text[:]) # 85
    print("bin sequence length", bin_sequence_length)
    while i <= bin_sequence_length:
        epoch_start_time = time.time()
        output, hidden = model(input, hidden)
        
        # print("bin: ", bin_)
        zero_index = zero[secret_text[:][i-1]]
        zero_index = torch.LongTensor(zero_index) 

        word_weights = output.squeeze().data.div(args.temperature).exp().cpu() 

        word_weights.index_fill_(0, zero_index, 0)
        word_idx = torch.multinomial(word_weights, 1)[0]
    
        input.data.fill_(word_idx)
        word = corpus.dictionary.idx2word[word_idx]

        if word not in common_tokens:
            i += 1
        w += 1
        word = word.encode('ascii', 'ignore').decode('ascii')
        outf.write(word + ('\n' if i % 20 == 19 else ' '))

        if i % args.log_interval == 0:
            print("total number of words", w)
            print("total length of secret", i)
            print('| Generated {}/{} words'.format(i, len(secret_text)))
            print('-' * 89)

