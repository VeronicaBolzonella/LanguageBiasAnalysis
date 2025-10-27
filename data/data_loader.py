import os
import re
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader

import nltk
from nltk.tokenize import sent_tokenize
nltk.download("punkt", quiet=True)

def preprocess_text(text):
    """Clean and split text into sentences."""
    text = re.sub(r"<.*?>", "", text) # Remove HTML tags
    text = re.sub(r"[^a-zA-Z0-9.,!?' ]", "", text) # Remove special characters
    text = re.sub(r"\s+", " ", text).strip() # Normalize spaces
    sentences = sent_tokenize(text) # Split into sentences
    return sentences


class BertEmbeddingDataset(Dataset):
    """Dataset returning tokenized sentences for BERT fine-tuning."""
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        # Remove batch dimension so DataLoader can handle it
        return {key: val.squeeze(0) for key, val in encoding.items()}


def load_books(folder):
    """Load all .txt files from the specified folder and preprocess them."""
    all_sentences = []
    for file in os.listdir(folder):
        if file.endswith(".txt") and not file == 'log.txt':
            with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                text = f.read()
                sentences = preprocess_text(text)
                all_sentences.extend(sentences)
    return all_sentences
