import re
import string
from enum import Enum
from collections import OrderedDict

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


stemmer = PorterStemmer()
stops = stopwords.words("english")

def pad(words: list, seq_length: int = 100):
    """
    Pad and/or Truncate text to seq_length
    """
    if len(words) < seq_length:
        words += ["<pad>"] * (seq_length - len(words))
    return words[:seq_length]


class Vocab:
    """
    Vocab object.
    """

    def __init__(self):
        self.word2id = OrderedDict({"<pad>": 0, "<unk>": 1})
        self.len = 2

    def __len__(self):
        return self.len

    def add_word(self, word: str):
        if not self.word2id.get(word):
            self.len += 1
            self.word2id[word] = self.len - 1

    def add_sent(self, sent):
        for word in sent.split():
            self.add_word(word)

    def get_index(self, word):
        id_ = self.word2id.get(word, None)
        if id_ is None:
            id_ = self.word2id["<unk>"]
        return id_

    def get_word(self, id_):
        return list(self.word2id.keys())[id_]


def preprocess(text: str) -> str:
    """
    Preprocess texts:
        - Convert to lowercase
        - Remove extra spaces
        - Remove stopwords, punctuations and non-alpha texts
        - stem
    """
    text = text.lower()
    text = re.sub("\s+", " ", text)
    clean_text = []
    for word in text.split():
        if (word not in stops) and (word not in string.punctuation) and word.isalpha():
            clean_text.append(stemmer.stem(word))
    return " ".join(clean_text)


class Modes(Enum):
    """
        Modes for app
    """
    CLI = "cli"
    GRADIO = "gradio"