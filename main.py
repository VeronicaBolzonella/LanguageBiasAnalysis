from __future__ import print_function, division
import bias_analysis.we as we
from bias_analysis.bias_plotting import plot_profession_bias
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from gensim.models import KeyedVectors
import os


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("embedding_filename", help="The name of the embedding")
    parser.add_argument("bias", help="bias axes between age, gender, class")
    parser.add_argument("target", help="word group between professions, animals, adj")

    args = parser.parse_args()

    if args.bias == "age":
        bias = "Age"
        direction_file = "data/terms/age_equalize_pairs.json"
    elif args.bias == "gender":
        bias = "Gender"
        direction_file = "data/terms/gender_equalize_pairs.json"
    elif args.bias == "class": 
        bias = "Class"
        direction_file = "data/terms/class_equalize_pairs.json"

    if args.target == "adj":
        target_file = "data/terms/adjectives.json"
        target = "Adjectives"
    elif args.target == "professions":
        target_file = "data/terms/professions.json"
        target = "Profession"
    elif args.target == "animals": 
        target_file = "data/terms/animals.json"
        target = "Animals"
    else:
        raise ValueError

    with open(direction_file, "r") as f:
        defs = json.load(f)

    E = KeyedVectors.load(args.embedding_filename)
    axis = we.doPCA(defs, E).components_[0]


    plot_profession_bias(E, axis, target_file, bias)