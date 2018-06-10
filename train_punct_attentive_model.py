from configs.punct_attentive_cfg import create_configuration
from utils import batchnize_dataset
from models.punct_attentive_model import SequenceLabelModel

print("Build configurations...")
config = create_configuration()

print("Load datasets...")
# used for training
train_set = batchnize_dataset(config["train_set"], config["batch_size"], shuffle=True)
# used for computing validate loss
valid_data = batchnize_dataset(config["dev_set"], batch_size=1000, shuffle=True)[0]
valid_text = config["dev_text"]
test_texts = [config["ref_text"], config["asr_text"]]

print("Build models...")
model = SequenceLabelModel(config)
model.train(train_set, valid_data, valid_text, test_texts)
