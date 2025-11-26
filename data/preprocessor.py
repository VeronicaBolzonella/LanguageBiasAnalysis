import os
import re
import pickle
import nltk
from nltk import sent_tokenize, word_tokenize

# nltk_data_dir = '/vol/tensusers/vbolzonella/txmm'
# os.makedirs(nltk_data_dir, exist_ok=True)
# nltk.data.path.append(nltk_data_dir)
# nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
# nltk.download('punkt_tab', download_dir=nltk_data_dir, quiet=True)

# # Add to NLTK search paths
# nltk.data.path.append(nltk_data_dir)

nltk.download('punkt')
nltk.download('punkt_tab')

INPUT_DIR = "data/gutenberg_children"
OUTPUT_FILE = "data/preprocessed.pkl"


def clean_text(text):
    # lowercase
    text = text.lower()

    # remove artifacts
    text = re.sub(r'\*{3}.*?\*{3}', ' ', text)

    # remove punctuation except apostrophes inside words
    text = re.sub(r"[^a-z0-9'\s]", " ", text)

    # normalize whitespace (not sure if necessary)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_file(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    cleaned = clean_text(raw)
    # tokenize sentences
    sents = sent_tokenize(cleaned)
    tokenized = [word_tokenize(s) for s in sents]
    tokenized = [t for t in tokenized if len(t) > 0]

    return tokenized


def preprocess_all_books():
    all_sentences = []

    for filename in os.listdir(INPUT_DIR):
        path = os.path.join(INPUT_DIR, filename)
        print(f"Processing {filename}...")

        sentences = preprocess_file(path)
        all_sentences.extend(sentences)

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(all_sentences, f)

    print(f"Saved preprocessed data to {OUTPUT_FILE}")


if __name__ == "__main__":
    preprocess_all_books()
