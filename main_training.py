import argparse
import os
import nltk

# nltk_data_dir = '/vol/tensusers/vbolzonella/txmm'
# os.makedirs(nltk_data_dir, exist_ok=True)
# nltk.data.path.append(nltk_data_dir)
# nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)

nltk.download('punkt')
# nltk.download('punkt_tab')

import numpy as np
import pickle

from training.fine_tuner import MyCorpus
from training.fine_tuner import train_word2vec

def main(epochs=10, vector_size:int=300, workers=8, folder:str="data/gutenberg_children"):
    sentences = MyCorpus(folder)

    # Train
    model = train_word2vec(sentences=sentences, vector_size=vector_size, epochs=epochs, workers=workers)

    # Get embedding matrix
    vectors = model.wv.vectors
    vocab = model.wv.index_to_key

    print(vectors.shape)
    print("Most similar to book: ", model.wv.most_similar("book", topn=3))
    print("Most similar to home: ", model.wv.most_similar("home", topn=5))
    print("Most similar to boy: ", model.wv.most_similar("boy", topn=5))

    # Save vectors
    model.wv.save("embeddings.kv")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, default="data/gutenberg_children", help="Path to the dataset")
    parser.add_argument("--vector_size", type=int, default=100, help="Size of embeddings")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--workers", type=int, default=15, help="CPU cores to use")
    args = parser.parse_args()

    main(epochs=args.epochs, vector_size=args.vector_size, folder=args.data_folder, workers=args.workers)
