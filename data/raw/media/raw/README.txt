This archive contains the training, development and testing files for
the MEDIA experiments reported in

@inproceedings{Vukotic.etal_2015,
  author = {Vedran Vukotic and Christian Raymond and Guillaume Gravier},
  title = {Is it time to switch to Word Embedding and Recurrent Neural Networks for Spoken Language Understanding?},
  booktitle = {InterSpeech},
  year = {2015},
  month = {September},
  address = {Dresde, Germany}
}

Files contains manual speech transciption of MEDIA without capitalization


all files are in wapiti/crf++ format:

one word per line
one empty line separating each utterance

files contain 3 columns:

1 : word itself
2 : word-class (done manually by myself (Christian Raymond)), by example XVILLE is the CITY_NAME class (sorry in French :))
3 : the label for the corresponding word using the BIO scheme to model concept segmentation