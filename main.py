import argparse
import os
import nltk

nltk_data_dir = '/vol/tensusers/vbolzonella/txmm'
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)
nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)

import numpy as np
import torch
import pickle
from gensim.models import KeyedVectors

from training.fine_tuner import train_word2vec, extract_embeddings
from data.data_loader import load_data

def main(epochs=3, vector_size:int=300, workers=8, folder:str="data/preprocessed_sentences.pkl"):
    # Load preprocessed data
    with open(folder, "rb") as f:
        data = pickle.load(f)

    # Train
    model = train_word2vec(data, vector_size=vector_size, epochs=epochs, workers=workers)

    # Get embedding matrix
    vectors, vocab = extract_embeddings(model)

    print(vectors.shape)
    print(model.wv.most_similar("home", topn=5))

    # Save vectors
    model.wv.save("embeddings.kv")

    # to load do
    # vectors = KeyedVectors.load("word_vectors.kv")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, default="data/preprocessed_sentences.pkl", help="Path to the dataset")
    parser.add_argument("--vector_size", type=int, default=300, help="Size of embeddings")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for training")
    parser.add_argument("--workers", type=int, default=8, help="CPU cores to use")
    args = parser.parse_args()
    
    main(epochs=args.epochs, vector_size=args.vector_size, folder=args.data_folder, workers=args.workers)
