from models.multi_attention_model import SequenceLabelModel
from utils import batchnize_dataset, load_dataset
from configs.media_multi_attention_cfg import create_configuration

print("Build configurations...")
config = create_configuration()

print("Load datasets...")
train_data = load_dataset(config["train_set"])
valid_set = batchnize_dataset(config["dev_set"], config["batch_size"], shuffle=False)
test_set = batchnize_dataset(config["test_set"], config["batch_size"], shuffle=False)
valid_data = batchnize_dataset(config["dev_set"], shuffle=False)

print("Build models...")
model = SequenceLabelModel(config)
model.train(train_data, valid_data, valid_set, test_set)

print("Inference...")
sentences = ["alors une nuit le DIZAINE MOIS UNITE MILLE UNITE", "dans un hôtel à XVILLE dans le centre ville"]
ground_truths = ["O B-sejour-nbNuit I-sejour-nbNuit B-temps-date I-temps-date I-temps-date B-temps-annee "
                 "I-temps-annee I-temps-annee",
                 "B-objetBD I-objetBD I-objetBD B-localisation-ville I-localisation-ville "
                 "B-localisation-lieuRelatif-general I-localisation-lieuRelatif-general "
                 "I-localisation-lieuRelatif-general I-localisation-lieuRelatif-general"]
for sentence, truth in zip(sentences, ground_truths):
    result = model.inference(sentence)
    print(result)
    print("Ground truth:\n{}\n".format(truth))
