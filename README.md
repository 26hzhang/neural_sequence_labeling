# Neural Sequence Labeling

![Authour](https://img.shields.io/badge/Author-Zhang%20Hao%20(Isaac%20Changhau)-blue.svg) ![](https://img.shields.io/badge/MacOS%20High%20Sierra-10.13.2-green.svg) ![](https://img.shields.io/badge/Python-3.6-brightgreen.svg) ![](https://img.shields.io/badge/TensorFlow-1.4.0-yellowgreen.svg)

A TensorFlow implementation of Neural Sequence Labeling model, which is able to tackle Part-of-Speech (POS) Tagging, Chunking and Named Entity Recognition (NER) tasks. This repository is inspired after reading the following papers about sequence labeling:
- [Named Entity Recognition with Bidirectional LSTM-CNNs](https://arxiv.org/pdf/1511.08308.pdf)
- [Neural Models for Sequence Chunking](https://arxiv.org/abs/1701.04027)
- [End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF](https://arxiv.org/pdf/1603.01354.pdf)
- [Neural Architectures for Named Entity Recognition](https://arxiv.org/pdf/1603.01360.pdf)
- [Optimal Hyperparameters for Deep LSTM-Networks for Sequence Labeling Tasks](https://arxiv.org/pdf/1707.06799.pdf)
- [Part-of-Speech Tagging from 97% to 100%: Is It Time for Some Linguistics?](https://nlp.stanford.edu/pubs/CICLing2011-manning-tagging.pdf)
- [Part-of-Speech Tagging with Bidirectional Long Short-Term Memory Recurrent Neural Network](https://arxiv.org/pdf/1510.06168.pdf)

This model follows the structure of `LSTM + CRF + Chars Embeddings (RNN/CNN)`, and several variant modules are available, like single bi-direct dynamic rnn, stacked bi-direct dynamic rnn and complex stacked bi-direct rnn (enhance the usage of char embeddings). RNN cell can use LSTM or GRU, and etc.

The performance (**F1 Score**) of this neural sequence labeling model on *NER* task is ***90.00~91.00***, which is near to the state-of-the-art performance (highest reported: `F1 Score = 91.21`, [ref. link](https://www.quora.com/What-is-the-current-state-of-the-art-in-Named-Entity-Recognition-NER)).

### Task

Given a sentence, give a tag to each word. For example, a classical application is Part-of-Speech (POS) Tagging
```bash
EU  rejects German call to boycott British lamb . 
NNP VBZ     JJ     NN   TO VB      JJ      NN   . 
```

For Chunking task
```bash
EU   rejects German call to   boycott British lamb . 
B-NP B-VP    B-NP   I-NP B-VP I-VP    B-NP    I-NP O 
```

For Named Entity Recognition (NER) task
```bash
Stanford University located at California 
B-ORG    I-ORG      O       O  B-LOC      
```

### Usage

**Note**: CoNLL 2003 English Dataset is obtained from [anago/data/conll2003/en/](https://github.com/Hironsan/anago/tree/master/data/conll2003/en), which is already placed in `data/conll2003/en` folder.

To download pre-trained word embeddings and pre-process data, run
```bash
# download embeddings (GloVe 6B and 840B word embeddings and GloVe 840B char embeddings)
$ ./download.sh
# pre-process data
$ python3 build_data.py 
```

Hyperparameters are stored in `config.py` (change to tasks here, train POS, Chunking or NER). To run the model, run
```bash
# training and testing model
$ python3 train_test.py
# if pretrained model exists in ckpt folder, restore and test
$ python3 restore_test.py
```

If training is processing normally, the log information will display as following
```bash
2018-01-10 13:51:26,225:INFO: Start training...
2018-01-10 13:51:27,771:INFO: Epoch  1/15:
2018-01-10 14:02:52,775:INFO: Testing model over DEVELOPMENT dataset
2018-01-10 14:03:57,189:INFO: accuracy: 97.76 -- f1 score: 88.30
2018-01-10 14:04:00,044:INFO:  -- new best score: 88.29796158040433
2018-01-10 14:04:00,045:INFO: Epoch  2/15:
2018-01-10 14:15:30,468:INFO: Testing model over DEVELOPMENT dataset
2018-01-10 14:16:36,593:INFO: accuracy: 98.24 -- f1 score: 90.86
2018-01-10 14:16:40,383:INFO:  -- new best score: 90.85699839892138
...
2018-01-10 16:42:36,101:INFO: Epoch 15/15:
2018-01-10 16:52:51,042:INFO: Testing model over DEVELOPMENT dataset
2018-01-10 16:53:43,379:INFO: accuracy: 98.86 -- f1 score: 94.76
2018-01-10 16:53:45,933:INFO:  -- new best score: 94.10280137965844
2018-01-10 16:53:45,933:INFO: Training process done...
2018-01-10 16:53:45,933:INFO: Testing model over TEST dataset
2018-01-10 16:54:34,322:INFO: accuracy: 97.93 -- f1 score: 90.66
```

After training and testing, a interact module is activated to allow user to manually test some input text
```bash
input> Stanford University located at California
Stanford University located at California 
B-ORG    I-ORG      O       O  B-LOC      
input> China is one of the biggest country in the World
China is one of the biggest country in the World 
B-LOC O  O   O  O   O       O       O  O   B-MISC
```

### Resources

#### Embeddings
- [GloVe Embeddings (6B, 42B, 840B)](https://nlp.stanford.edu/projects/glove/)
- [minimaxir/char-embeddings](https://github.com/minimaxir/char-embeddings)
- [GloVe 840B 300 dimension Char Embeddings](https://github.com/minimaxir/char-embeddings/blob/master/glove.840B.300d-char.txt)

#### GitHub Repositories of Sequence Labeling (Ref.)
- [LopezGG/NN_NER_tensorFlow](https://github.com/LopezGG/NN_NER_tensorFlow), implementation of [End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF](https://arxiv.org/pdf/1603.01354.pdf)
- [UKPLab/emnlp2017-bilstm-cnn-crf](https://github.com/UKPLab/emnlp2017-bilstm-cnn-crf), implementation of [Reporting Score Distributions Makes a Difference: Performance Study of LSTM-networks for Sequence Tagging](https://arxiv.org/pdf/1707.09861.pdf) and [Optimal Hyperparameters for Deep LSTM-Networks for Sequence Labeling Tasks](https://arxiv.org/pdf/1707.06799.pdf)
- [ThanhChinhBK/Ner-BiLSTM-CNNs](https://github.com/ThanhChinhBK/Ner-BiLSTM-CNNs)
- [clab/stack-lstm-ner](https://github.com/clab/stack-lstm-ner), implementation of [Neural Architectures for Named Entity Recognition](http://arxiv.org/pdf/1603.01360v1.pdf)
- [Hironsan/anago](https://github.com/Hironsan/anago)
- [guillaumegenthial/sequence_tagging](https://github.com/guillaumegenthial/sequence_tagging), [blog](https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html)
- [glample/tagger](https://github.com/glample/tagger), implementation of [Neural Architectures for Named Entity Recognition](https://arxiv.org/pdf/1603.01360.pdf)

#### Others
- [models/tutorials/rnn/quickdraw/train_model.py](https://github.com/tensorflow/models/blob/master/tutorials/rnn/quickdraw/train_model.py)
