from __future__ import print_function, division
import bias_analysis.we as we
from bias_analysis.bias_plotting import plot_profession_bias, plot_adjectives_bias, plot_animal_bias
# from bias_analysis.bias_plotting_compare import plot_profession_bias
from bias_analysis.shift_evaluation import calculate_shift, plot_easy_shift
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from gensim.models import KeyedVectors
import os


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("embedding_filename", help="The name of the embedding")
    parser.add_argument("embedding_new_filename", help="The name of the new embedding")

    # parser.add_argument("bias", help="bias axes between age, gender, class")
    # parser.add_argument("target", help="word group between professions, animals, adj")

    args = parser.parse_args()
    
    age_direction_file = "data/terms/age_equalize_pairs.json"
    gender_direction_file = "data/terms/gender_equalize_pairs.json"
    class_direction_file = "data/terms/class_equalize_pairs.json"

    adj_target_file = "data/terms/sadjectives.json"
    prof_target_file = "data/terms/sprofessions.json"
    animals_target_file = "data/terms/sanimals.json"


    with open(age_direction_file, "r") as f:
        age_defs = json.load(f)

    with open(gender_direction_file, "r") as f:
        gender_defs = json.load(f)

    with open(class_direction_file, "r") as f:
        class_defs = json.load(f)

    E = KeyedVectors.load(args.embedding_filename)
    # E_new = KeyedVectors.load(args.embedding_new_filename)
    age_axis = we.doPCA(age_defs, E).components_[0]
    gender_axis = we.doPCA(gender_defs, E).components_[0]
    class_axis = we.doPCA(class_defs, E).components_[0]
    print(E.vector_size)
    # print(E_new.vector_size)

    bias_axes = [age_axis, gender_axis, class_axis]
    bias = ["Gender", "Age", "Class"]

    sets = zip(bias_axes, bias)

    for axis, bias in sets:
        revert = True
        scale = 1
        if bias=="Class":
            revert=False
            scale=-1
        if bias=="Age":
            revert=False

        # plot_animal_bias(E, axis, animals_target_file, bias, -revert, scale)
        plot_profession_bias(E=E, axis=axis, target_file=prof_target_file, bias=bias, revert=-revert, scale=scale)
        # plot_adjectives_bias(E, axis, adj_target_file, bias, -revert, scale)

    # for axis, bias in sets:
    #     calculate_shift(E, E_new, axis, prof_target_file, bias, "Professions")
    #     calculate_shift(E, E_new, axis, adj_target_file, bias, "Adjectives")        
        # calculate_shift(E, E_new, axis, animals_target_file, bias, "Animals")
