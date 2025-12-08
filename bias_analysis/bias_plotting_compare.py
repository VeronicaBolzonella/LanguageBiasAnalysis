from __future__ import print_function, division
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

def legend_handles(categories, colors):
    unique_legend = {}
    for name, col in zip(categories, colors):
        unique_legend[name] = col
    # Legend
    return [plt.Rectangle((0, 0), 1, 1, fc=col, label=name)
                      for name, col in unique_legend.items()]

def projection_on_axis(E, word, axis):
    v = E.get_vector(word)
    return np.dot(v, axis) / (np.linalg.norm(v) * np.linalg.norm(axis))

def plot_profession_bias(E, E_new, axis, target_file, bias, revert, scale=1):
    category_map = {
            "S": ("STEM", "#1f77b4"),
            "H": ("Doctors", "#ff7f0e"),
            "B": ("Business & Finance", "#2ca02c"),
            "P": ("Government & Law", "#d62728"),
            "E": ("Education and Academia", "#9467bd"),
            "A": ("Arts, Design & Media", "#8c564b"),
            "F": ("Service & Support", "#e377c2"),
            "C": ("Caregiving", "#7f7f7f")
        }
    
    with open(target_file, "r") as f:
        # [(w, c)x N]
        target_words = json.load(f)


    w2c = {w: category_map[c] for w, c in target_words if w in E} 
    # eg 'writer': ('Arts, Design & Media', '#8c564b')

    target_scores = {w: scale*projection_on_axis(E, w, axis) for w in w2c}
    target_scores_new = {w: scale*projection_on_axis(E_new, w, axis) for w in w2c}

    sorted_targets = sorted(target_scores.items(), key=lambda x: x[1], reverse=revert)
    words, scores = zip(*sorted_targets)

    sorted_words = sorted(target_scores.keys(), key=lambda w: target_scores[w], reverse=revert)
    vals1 = [target_scores[w] for w in sorted_words]
    vals2 = [target_scores_new[w] for	w in sorted_words]
    colors = [w2c[w][1] for w in sorted_words]

    # colors = []
    # colors = ["#ff7f0e", "#7f7f7f", "#1f77b4", "#e377c2", "#2ca02c", "#9467bd", "#8c564b"]
    # plot_filtered(words, scores, w2c, colors, bias, target="Professions")


    for code, (cat_name, cat_color) in category_map.items():
        # select words belonging to this category
        words_in_cat = [w for w, c in target_words if w in w2c and category_map[c][0] == cat_name]

        if not words_in_cat:
            continue

        # sort them consistently
        sorted_words = sorted(words_in_cat, key=lambda w: target_scores[w], reverse=revert)

        vals1  = [target_scores[w]     for w in sorted_words]
        vals2  = [target_scores_new[w] for w in sorted_words]

        # plot the comparison for this category
        plot_compare(sorted_words, vals1, vals2, bias, f"Professions_{cat_name}")



def get_axis_name(bias):
    if bias == "Age":
        return "from Young to Old"
    elif bias == "Gender":
        return "from He to She"
    elif bias == "Class": 
        return "from Poor to Rich"
    

def plot_compare(words, vals1, vals2, bias, target):
    # width offsets
    x = np.arange(len(words))
    width = 0.4

    plt.figure(figsize=(15, 6))

    plt.bar(x - width/2, vals1, width=width, color="orange", alpha=0.6, label="Original")
    plt.bar(x + width/2, vals2, width=width, color="blue", alpha=0.9, label="Finetuned")

    plt.xticks(x, words, rotation=90)

    axis_name = get_axis_name(bias)
    plt.ylabel(f"Projection on {bias} Bias Axis")
    plt.xlabel(f"Word (sorted {axis_name})")
    plt.title(f"Comparison of E1 vs E2 on {bias} axis")

    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(f"data/results/{bias}_{target}_comparison.png", dpi=300, bbox_inches="tight")
