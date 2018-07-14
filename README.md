# Neural Sequence Labeling

![Authour](https://img.shields.io/badge/Author-Zhang%20Hao%20(Isaac%20Changhau)-blue.svg) ![Python](https://img.shields.io/badge/Python-3.6.5-brightgreen.svg) ![Tensorflow](https://img.shields.io/badge/TensorFlow-1.8.0-yellowgreen.svg)

A TensorFlow implementation of Neural Sequence Labeling model, which is able to tackle sequence labeling tasks such as _Part-of-Speech (POS) Tagging_, _Chunking_, _Named Entity Recognition (NER)_, _Punctuation Restoration_, _Sentence Boundary Detection_, _Spoken Language Understanding_ and so forth.

## Tasks

> All the models are trained by one GeForce GTX 1080Ti GPU.

### Part-Of-Speech (POS) Tagging Task
Experiment on `CoNLL 2003 POS dataset`, (**45** target annotations), example:
```bash
EU  rejects German call to boycott British lamb . 
NNP VBZ     JJ     NN   TO VB      JJ      NN   . 
```
This task is a typical sequence labeling, and current SOTA POS tagging methods are well solved this problem, those methods work rapidly and reliably, with per-token accuracies of slightly over **97%** ([ref.](https://nlp.stanford.edu/pubs/CICLing2011-manning-tagging.pdf)). So, for the model of this task, I just follow the structure of ***NER Task***, to build a POS tagger, which is able to achieve good performance.

Similarly, all the configurations are put in the [train_conll_pos_blstm_cnn_crf.py](train_conll_pos_blstm_cnn_crf.py) and the model is built in the [blstm_cnn_crf_model.py](/models/blstm_cnn_crf_model.py). Simply run `python3 train_conll_pos_blstm_cnn_crf.py` to start a training process.

### Chunking Task
Experiment on `CoNLL 2003 Chunk dataset`, (**21** target annotations), example:
```bash
EU   rejects German call to   boycott British lamb . 
B-NP B-VP    B-NP   I-NP B-VP I-VP    B-NP    I-NP O 
```
This task is also similar to the NER task below, so the model for Chunking task also follows the structure of **NER**. ALL the configurations are put in the [train_conll_chunk_blstm_cnn_crf.py](train_conll_chunk_blstm_cnn_crf.py) and the model is built in the [blstm_cnn_crf_model.py](/models/blstm_cnn_crf_model.py).

To achieve the SOTA results, the parameters are need to be carefully tuned.

### Named Entity Recognition (NER) Task
Experiment on `CoNLL 2003 NER dataset`, standard `BIO2` annotation format (**9** target annotations), example:
```bash
Stanford University located at California .
B-ORG    I-ORG      O       O  B-LOC      O
```
To tackle this task, I build the model follows the structure of `Words Embeddings + Chars Embeddings (RNNs/CNNs) + RNNs + CRF` (basement), and several variant modules as well as attention mechanism are available. All the configurations are put in the [train_conll_ner_blstm_cnn_crf.py](train_conll_ner_blstm_cnn_crf.py) and the model is built in the [blstm_cnn_crf_model.py](/models/blstm_cnn_crf_model.py).

The SOTA performance (F1 score, [ref](https://www.quora.com/What-is-the-current-state-of-the-art-in-Named-Entity-Recognition-NER)) is `F1 Score = 91.21` achieved by [End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF](https://arxiv.org/pdf/1603.01354.pdf), so the basement model also follow the similar structure as this paper, but the parameters setting is different.

To train and inference the model, directly run `python3 train_conll_ner_blstm_cnn_crf.py`, and below gives an example of training basement model and achieves `F1 Score = 91.82`, which is similar to the SOTA results.

> Unlike the `End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF`, the basement model converges much faster than this SOTA method, since the basement model (with large embedding dimension and rnn hidden units) has much more parameters than the SOTA method, although, the basement model also has high probability to be overfitting.

```bash
Build models...
word + chars embedding shape: [None, None, 500]
rnn output shape: [None, None, 600]
logits shape: [None, None, 9]
Start training...
...
Epoch 47/100:
703/703 [==============================] - 95s - Global Step: 33041 - Train Loss: 0.0246      
Valid dataset -- accuracy: 98.53, precision: 91.54, recall: 92.12, FB1: 91.83
 -- new BEST score on valid dataset: 91.83
Test dataset -- accuracy: 98.52, precision: 91.53, recall: 92.11, FB1: 91.82
```
Some SOTA NER F1 score on test data set from CoNLL-2003:

| Model | F1 Score (on CoNLL 2003 dataset) |
| :---  | :---: |
| [Bidirectional LSTM-CRF Models for Sequence Tagging](https://arxiv.org/pdf/1508.01991.pdf) | 90.10 |
| [Neural Architectures for Named Entity Recognition](https://arxiv.org/pdf/1603.01360.pdf) | 90.94 |
| [Multi-Task Cross-Lingual Sequence Tagging from Scratch](https://arxiv.org/pdf/1603.06270.pdf) | 91.20 |
| [End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF](https://arxiv.org/pdf/1603.01354.pdf) | 91.21 |
| The Basement Model | 91.82 (+0.11, -0.19) |

> Some reported F1 scores are higher than 91.21 are not shown here, since the author of `End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF` mentioned that they are not comparable to the work since they use larger data set for training.

The variant modules include Stack Bidirectional RNN (multi-layers), Multi-RNN Cells (multi-layers), Lurong/Bahdanau Attention Mechanism, Self-attention Mechanism, Residual Connection, Layer Normalization and so on. However, these modifications did not improve (sometime even worse than basement model) the performance significantly (`F1 score improves >= 1.5`). It's easy to apply those variants and train by modifying the config settings in [train_conll_ner_blstm_cnn_crf.py](train_conll_ner_blstm_cnn_crf.py).

### Spoken Language Understanding Task
Experiment on `MEDIA dataset` (French language), standard `BIO2` annotation format (**138** target annotations), example:
```bash
réserver        dans l'       un       de              ces             hôtels
B-command-tache O    B-nombre I-nombre B-lienRef-coRef I-lienRef-coRef B-objetBD
```

> Current highest F1 score is 86.4, which is slightly lower than the SOTA F1 result, which is 86.95.

Details of configurations and model are placed in [train_media_multi_attention.py](train_media_multi_attention.py) and [multi_attention_model.py](/models/multi_attention_model.py) separately.

### Punctuation Restoration/Sentence Boundary Detection Task
Experiment on `Transcripts of TED Talks (IWSLT) dataset`, example:
```bash
$ raw sentence: I'm a savant, or more precisely, a high-functioning autistic savant.
$ processed annotation format:
$ i 'm a savant or more precisely a high-functioning autistic savant
$ O O  O COMMA  O  O    COMMA     O O                O        PERIOD
```
To deal with this task, I build an attention-based model, which follows the structure `Words Embeddings + Chars Embeddings (RNNs/CNNs) + Densely Connected Bi-LSTM + Attention Mechanism + CRF`. All the configurations are put in the [train_punct_attentive_model.py](train_punct_attentive_model.py) and the model is built in the [punct_attentive_model.py](/models/punct_attentive_model.py).

To train the model, directly run `python3 train_punct_attentive_model.py`, and below gives an example of attentive model and achieves `F1 Score = 68.9`, which is slightly higher than the SOTA results.
```bash
Build models...
word embedding shape: [None, None, 300]
chars representation shape: [None, None, 100]
word and chars concatenation shape: [None, None, 400]
densely connected bi_rnn output shape: [None, None, 600]
attention output shape: [None, None, 300]
logits shape: [None, None, 4]
Start training...
...
Epoch 5/30:
349/349 [==============================] - 829s - Global Step: 1745 - Train Loss: 28.7219
Evaluate on data/raw/LREC_converted/ref.txt:
----------------------------------------------
PUNCTUATION      PRECISION RECALL    F-SCORE  
,COMMA           64.8      59.6      62.1     
.PERIOD          73.5      77.0      75.2     
?QUESTIONMARK    70.8      73.9      72.3     
----------------------------------------------
Overall          69.4      68.4      68.9
Err: 5.96%
SER: 44.8%
Evaluate on data/raw/LREC_converted/asr.txt:
----------------------------------------------
PUNCTUATION      PRECISION RECALL    F-SCORE  
,COMMA           49.7      53.5      51.5     
.PERIOD          67.5      70.9      69.2
?QUESTIONMARK    51.4      54.3      52.8
----------------------------------------------
Overall          58.4      62.1      60.2
Err: 8.23%
SER: 64.4%
```
Some SOTA scores on English reference transcripts (`ref`) and ASR output testset (`asr`) from IWSLT:  
![ref-asr](/assets/ref-asr.png)

> The overall F1 score of the attentive model on `ref` dataset is `67.6~69.5`, while on `asr` dataset is `60.2~61.5`.

## Resources
### Datasets
- [CoNLL 2003 POS, Chunking and NER datasets](https://github.com/synalp/NER/tree/master/corpus/CoNLL-2003).
- MEDIA dataset, details in [Is it time to switch to Word Embedding and Recurrent Neural Networks for Spoken Language Understanding?](https://hal.inria.fr/hal-01196915/document).
- Transcripts of TED Talks (IWSLT) dataset for Punctuation Restoration or Sentence Boundary Detection, details in [Bidirectional Recurrent Neural Network with Attention Mechanism for Punctuation Restoration](https://www.researchgate.net/publication/307889284_Bidirectional_Recurrent_Neural_Network_with_Attention_Mechanism_for_Punctuation_Restoration). I also provided the converted dataset built from the raw dataset, which can be used to train a model directly.

### Embeddings and Evaluation Script
- [GloVe Embeddings (6B, 42B, 840B)](https://nlp.stanford.edu/projects/glove/)
- [minimaxir: GloVe 840B 300d Char Embeddings](https://github.com/minimaxir/char-embeddings/blob/master/glove.840B.300d-char.txt)
- [Word2Vec Google News 300d Embeddings](https://github.com/mmihaltz/word2vec-GoogleNews-vectors)
- [AdolfVonKleist/rnn-slu/rnn-slu/CoNLLeval.py](https://github.com/AdolfVonKleist/rnn-slu/blob/master/rnnslu/CoNLLeval.py)

### Papers
- [Named Entity Recognition with Bidirectional LSTM-CNNs](https://arxiv.org/pdf/1511.08308.pdf)
- [Bidirectional LSTM-CRF Models for Sequence Tagging](https://arxiv.org/pdf/1508.01991.pdf)
- [End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF](https://arxiv.org/pdf/1603.01354.pdf)
- [Neural Architectures for Named Entity Recognition](https://arxiv.org/pdf/1603.01360.pdf)
- [Multi-Task Cross-Lingual Sequence Tagging from Scratch](https://arxiv.org/pdf/1603.06270.pdf)
- [Part-of-Speech Tagging from 97% to 100%: Is It Time for Some Linguistics?](https://nlp.stanford.edu/pubs/CICLing2011-manning-tagging.pdf)
- [Part-of-Speech Tagging with Bidirectional Long Short-Term Memory Recurrent Neural Network](https://arxiv.org/pdf/1510.06168.pdf)
- [Bidirectional Recurrent Neural Network with Attention Mechanism for Punctuation Restoration](https://pdfs.semanticscholar.org/8785/efdad2abc384d38e76a84fb96d19bbe788c1.pdf?_ga=2.156364859.1813940814.1518068648-1853451355.1518068648)

### Others
- [Difference between MultiRNNCell and stack_bidirectional_dynamic_rnn in Tensorflow](https://stackoverflow.com/questions/49242266/difference-between-multirnncell-and-stack-bidirectional-dynamic-rnn-in-tensorflo)
- [Chinese Sequence Labeling using Deep Learning Method](https://www.jianshu.com/p/7e233ef57cb6)
- [Building a large annotated corpus of English: the Penn Treebank](https://catalog.ldc.upenn.edu/docs/LDC95T7/cl93.html)
- [PennTree Bank POS tags](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html)
