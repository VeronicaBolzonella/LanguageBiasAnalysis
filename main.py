import argparse
import numpy as np
import torch
import pickle


from training.fine_tuner import train_word2vec, extract_embeddings
from data.data_loader import load_data

def main(epochs=3, vector_size:int=300, folder:str="data/preprocessed_sentences.pkl"):
    # Load preprocessed data
    with open(folder, "rb") as f:
        data = pickle.load(f)

    # Train
    model = train_word2vec(data, vector_size=vector_size, epochs=epochs)

    # Get embedding matrix
    vectors, vocab = extract_embeddings(model)

    print(vectors.shape)
    print(vocab[:20])
    print("Vector for 'home':", model.wv['home'][:10])
    print(model.wv.most_similar("home", topn=5))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, default="data/preprocessed_sentences.pkl", help="Path to the dataset")
    parser.add_argument("--vector_size", type=int, default=300, help="Size of embeddings")
    parser.add_argument("--epochs", type=int, default=8, help="Number of epochs for training")

    args = parser.parse_args()
    main(epochs=args.epochs, vector_size=args.vector_size, folder=args.data_folder)
