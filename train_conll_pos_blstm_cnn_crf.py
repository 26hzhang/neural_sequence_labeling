from models.blstm_cnn_crf_model import SequenceLabelModel
from utils import batchnize_dataset
from configs.conll_pos_blstm_cnn_crf_cfg import create_configuration


print("Build configurations...")
config = create_configuration()

print("Load datasets...")
# used for training
train_set = batchnize_dataset(config["train_set"], config["batch_size"], shuffle=True)
# used for computing validate loss
valid_data = batchnize_dataset(config["dev_set"], batch_size=1000, shuffle=True)[0]
# used for computing validate accuracy, precision, recall and F1 scores
valid_set = batchnize_dataset(config["dev_set"], config["batch_size"], shuffle=False)
# used for computing test accuracy, precision, recall and F1 scores
test_set = batchnize_dataset(config["test_set"], config["batch_size"], shuffle=False)

print("Build models...")
model = SequenceLabelModel(config)
model.train(train_set, valid_data, valid_set, test_set)

print("Inference...")
sentences = ["EU rejects German call to boycott British lamb ."]
ground_truths = ["NNP VBZ     JJ     NN   TO VB      JJ      NN   ."]
for sentence, truth in zip(sentences, ground_truths):
    result = model.inference(sentence)
    print(result)
    print("Ground truth:\n{}\n".format(truth))
