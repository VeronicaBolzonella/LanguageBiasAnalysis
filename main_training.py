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
from training.fine_tuner import train_word2vec, fine_tune_w2v

def main_train(epochs=10, vector_size:int=100, workers=12, folder:str="data/gutenberg_children_plus", output_filename:str="new_embeddings.kv"):
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
    model.wv.save("embeddings_new.kv")

def main_finetune(E:str="ebeddings_local.kv", epochs=10, workers=15, folder:str="data/freestories_children"):
    sentences = MyCorpus(folder)

    # Train
    model = fine_tune_w2v(E, new_sentences=sentences, epochs=epochs)

    # Get embedding matrix
    vectors = model.wv.vectors

    print(vectors.shape)
    print("Most similar to book: ", model.wv.most_similar("book", topn=3))
    print("Most similar to home: ", model.wv.most_similar("home", topn=5))
    print("Most similar to boy: ", model.wv.most_similar("boy", topn=5))

    # Save vectors
    model.wv.save("embeddings.kv")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_dir", type=str, default="embeddings_local.kv", help="Path to the embeddings")
    parser.add_argument("--data_folder", type=str, default="data/gutenberg_children_plus", help="Path to the dataset")
    parser.add_argument("--epochs", type=int, default=8, help="Number of epochs for training")
    args = parser.parse_args()

    main_train(epochs=args.epochs, folder=args.data_folder)
