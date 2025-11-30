from we import WordEmbedding
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from gensim.models import KeyedVectors



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("embedding_filename", help="The name of the embedding")
    parser.add_argument("bias", help="bias axes between age, gender, class")
    # parser.add_argument("target", help="word group between prfessions, animals, adj")

    args = parser.parse_args()

    if args.bias == "age":
        v_words = ("young", "old")
    elif args.bias == "gender":
        v_words = ("she", "he")
    elif args.bias == "class": 
        v_words = ("poor", "rich")
    else:
        raise ValueError

    w = WordEmbedding(args.embedding_filename)

    # Convert to vector difference
    try:
        v = w[v_words[0]] - w[v_words[1]]   # vector difference
    except KeyError as e:
        raise ValueError(f"Word not in embedding: {e}")

    
    analogies = w.best_analogies_dist_thresh(v, topn=50)

    print(analogies)