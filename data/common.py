import codecs
import ujson
import re
import unicodedata
from unicodedata import normalize

PAD = "<PAD>"
UNK = "<UNK>"
NUM = "<NUM>"
END = "</S>"
SPACE = "_SPACE"


def write_json(filename, dataset):
    with codecs.open(filename, mode="w", encoding="utf-8") as f:
        ujson.dump(dataset, f)


def word_convert(word, language="english", keep_number=True, lowercase=True):
    # convert french characters to latin equivalents
    if language.lower() == "french":
        word = normalize("NFD", word).encode("ascii", "ignore").decode("utf-8")
    if not keep_number:
        if is_digit(word):
            word = NUM
    if lowercase:
        word = word.lower()
    return word


def is_digit(word):
    try:
        float(word)
        return True
    except ValueError:
        pass
    try:
        unicodedata.numeric(word)
        return True
    except (TypeError, ValueError):
        pass
    result = re.compile(r'^[-+]?[0-9]+,[0-9]+$').match(word)
    if result:
        return True
    return False
