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
        v = ("young", "old")
    elif args.bias == "gender":
        v = ("she", "he")
    elif args.bias == "class": 
        v = ("poor", "rich")
    else:
        raise ValueError
    
    w = WordEmbedding(args.embedding_filename)

    analogies = w.best_analogies_dist_thresh(v)

    print(analogies)