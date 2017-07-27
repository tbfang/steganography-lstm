# Evaluate Test Perplexity given the model or secret text
# Average probabilities

import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import sys
import random
import numpy as np
import itertools

import data
import model

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/small-enron',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='./models/small-enron-model.pt',
                    help='model path')
parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=20,
                    help='sequence length')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=3000,
                    help='reporting interval')
parser.add_argument('--bins', type=int, default=1,
                    help='number of word bins')
parser.add_argument('--replication_factor', type=int, default=1,
                    help='number of words each word appears')
parser.add_argument('--common_bin_factor', type=int, default=0,
                    help='how many bins to add common words')
parser.add_argument('--num_tokens', type=int, default=0,
                    help='common words is n tokens')
parser.add_argument('--threshold', type=int, default=1, metavar='N',
                    help='# of words a word must occur to be included in dictionary')
parser.add_argument('--experiments', type=int, default=1, metavar='N',
                    help='# of times to run the evaluate')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data

eval_batch_size = 10
# train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
# test_data = batchify(corpus.test, eval_batch_size)

print('Loaded data')

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary.word_count)
print("vocabulary size:", ntokens)

with open(args.model, 'rb') as f:
    model = torch.load(f)

criterion = nn.NLLLoss()

print('Loaded model')

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_batch(source, i, evaluation=False):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target

###############################################################################
# Evaluation
###############################################################################

def get_common_tokens(n):
    dictionary = corpus.dictionary.word_count
    d = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)
    common_tokens = [item[0] for item in d]
    common_tokens = common_tokens[0:n]
    return common_tokens

if args.bins >= 2:
    sub_bin_indices = np.random.choice(range(args.bins), size=args.replication_factor, replace=False)
    common_bin_indices = np.random.choice(range(args.bins), size=args.common_bin_factor, replace=False) 

    ntokens = len(corpus.dictionary) 
    tokens = list(range(ntokens)) # * args.replication_factor

    random.shuffle(tokens)
    words_in_bin = int(len(tokens) / args.bins) 

    # common words
    common_tokens = get_common_tokens(args.num_tokens)
    remove_words = ['<user>','rt']
    common_tokens = list(set(common_tokens) - set(remove_words))
    common_tokens_idx = [corpus.dictionary.word2idx[word] for word in common_tokens]

    bins = [tokens[i:i + words_in_bin] for i in range(0, len(tokens), words_in_bin)] # words to keep in each bin...
    bins = [list(set(tokens) - set(bin_)) for bin_ in bins]

    sub_bins = [bins[index] for index in sub_bin_indices]
    replicated_bin = list(itertools.chain(*sub_bins)) # just one bin

    # print("size of bins before common words", len(bins[0]))
    bins = [replicated_bin if bins.index(bin_) in sub_bin_indices else bin_ for bin_ in bins]
    bins = [list(set(bin_) - set(common_tokens_idx)) if bins.index(bin_) in common_bin_indices else bin_ for bin_ in bins]
    # print("size of bins after common words", len(bins[0]))

def log_softmax(unnormalized_probs, bin_):
    denom = torch.sum(unnormalized_probs.exp(), 1) # denom is a 200 * 1 tensor
    denom = denom.expand(denom.size(0), unnormalized_probs.size(1))
    probs = torch.div(unnormalized_probs.exp(), denom)
    if args.bins >= 2:
        # print("before masking probabilities", torch.sum(probs.data, 1))
        mask_probabilities(probs, bin_)
        # print("after masking probabilities", torch.sum(probs.data, 1))
    log_probs = torch.log(probs)
    return log_probs # output is a n * vocab tensor

# instead of filling 0, fill the average probability. :)
def mask_probabilities(probs, bin_):
    mask_words = bins[bin_]
    mask_words = list(set(mask_words))

    divided_probs = torch.div(probs, args.bins)
    numpy_divided_probs = divided_probs.cpu().data.numpy()
    numpy_probs = probs.cpu().data.numpy()
    numpy_probs[:,mask_words] = numpy_divided_probs[:,mask_words]
    probs.data = torch.FloatTensor(numpy_probs).cuda()

def evaluate(data_source):
    total_loss = 0
    ntokens = len(corpus.dictionary.word_count)
    hidden = model.init_hidden(eval_batch_size)
    for i in range(0, data_source.size(0), args.bptt):
        data, targets = get_batch(data_source, i, evaluation=True)
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        bin_ = np.random.choice(range(args.bins))
        output_flat = log_softmax(output_flat, bin_)
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
        if i % args.log_interval == 0:
            print('| Generated {}/{} words'.format(i, ntokens))
            print('-' * 89)
    return total_loss[0] / len(data_source)

###############################################################################
# Run
###############################################################################

# Run on test data and save the model.

valid_loss = 0
for i in range(args.experiments):
    valid_loss += evaluate(val_data)
valid_loss = valid_loss / args.experiments
print('=' * 89)
print('| End of training | valid loss {:5.2f} | valid ppl {:8.2f}'.format(
    valid_loss, math.exp(valid_loss)))
print('=' * 89)
