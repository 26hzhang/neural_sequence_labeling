#!/usr/bin/env bash

# Download GloVe 840B
DATA_DIR=./data/
GLOVE840_DIR=${DATA_DIR}/glove.840B
mkdir -p ${GLOVE840_DIR}
wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O ${GLOVE840_DIR}/glove.840B.300d.zip
unzip ${GLOVE840_DIR}/glove.840B.300d.zip -d ${GLOVE840_DIR}
# Download Glove Character Embedding
wget https://raw.githubusercontent.com/minimaxir/char-embeddings/master/glove.840B.300d-char.txt -O ${GLOVE840_DIR}/glove.840B.300d-char.txt

# Download GloVe 6B
GLOVE6_DIR=${DATA_DIR}/glove.6B
wget http://nlp.stanford.edu/data/glove.6B.zip -O ${GLOVE6_DIR}/glove.6B.zip
unzip ${GLOVE6_DIR}/glove.6B.zip -d ${GLOVE6_DIR}
