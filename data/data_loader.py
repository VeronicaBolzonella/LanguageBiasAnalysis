import os
import re
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader

import nltk
from nltk import sent_tokenize, word_tokenize

# nltk_data_dir = '/vol/tensusers/vbolzonella/txmm'
# os.makedirs(nltk_data_dir, exist_ok=True)
# nltk.data.path.append(nltk_data_dir)
# nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
# nltk.download('punkt_tab', download_dir=nltk_data_dir, quiet=True)

nltk.download('punkt')
nltk.download('punkt_tab')


def preprocess_text(text):
    """Clean and split text into sentences."""
    text = re.sub(r"<.*?>", "", text) # Remove HTML tags
    text = re.sub(r"[^a-zA-Z0-9.,!?' ]", "", text) # Remove special characters
    text = re.sub(r"\s+", " ", text).strip() # Normalize spaces
    sentences = sent_tokenize(text) 
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
    

def load_data(batch_size=8, folder:str="data/gutenberg_children"):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    all_sentences = []
    for file in os.listdir(folder):
        if file.endswith(".txt") and not file == 'log.txt':
            with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                text = f.read()
                sentences = preprocess_text(text)
                all_sentences.extend(sentences)

    dataset = BertEmbeddingDataset(all_sentences, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader