# Steganography LSTM

Code for the paper [Generating Steganographic Text with LSTMs](https://arxiv.org/abs/1705.10742). The LSTM is based on the [Word Language Model](https://github.com/pytorch/examples/tree/master/word_language_model) example from PyTorch (http://pytorch.org/).

## Requirements

- Latest [NVIDIA driver](http://www.nvidia.com/Download/index.aspx)
- [CUDA 8 Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [cuDNN](https://developer.nvidia.com/cudnn)
- [PyTorch](https://github.com/pytorch/pytorch#installation)

## Data
- A small sample of Penn Treebank and Tweets. `pre-process.py` is tokenization of punctuation.

## Training
- `python main.py --cuda --nhid 600 --nlayers 3 --epochs 6 --data './data/tweets --save './models/twitter-model.pt'`
For the full list of arguments, check the [PyTorch example README](https://github.com/pytorch/examples/tree/master/word_language_model).

## Text Generation
One of our key and original contributions. After we train our model, we generate words and restrict the output based on the secret text. `generate.py` is modified such that it takes the secret text and modifies the probabilities based on the "bins" as described in our paper.
Example generation with 4 bins: 
` python generate.py --data './data/tweets' --checkpoint './models/twitter-model.pt' --cuda --outf 'outputs/stegotweets.txt' --words 1000 --temperature 0.8 --bins 4 --common_bin_factor 4 --num_tokens 20`
See the arguments in `generate.py` or refer to the [PyTorch example README](https://github.com/pytorch/examples/tree/master/word_language_model).

## Evaluation
We proposed and implemented an alternate measure of perplexity in Section 3.2 of [our paper](https://arxiv.org/abs/1705.10742). The code is in `evaluate.py`.

Example evaluation: `python evaluate.py --cuda --data './data/tweets' --model './models/twitter-model.pt' --bins 4`

If there are any questions or concerns about the code or paper, please contact Tina Fang at tbfang@edu.uwaterloo.ca. We would love to hear your feedback!