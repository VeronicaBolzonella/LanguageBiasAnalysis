from __future__ import print_function, division
import json
import numpy as np
import matplotlib.pyplot as plt
import os

def legent_handles(categories, colors):
    unique_legend = {}
    for name, col in zip(categories, colors):
        unique_legend[name] = col
    # Legend
    return [plt.Rectangle((0, 0), 1, 1, fc=col, label=name)
                      for name, col in unique_legend.items()]

def projection_on_axis(E, word, axis):
            v = E.get_vector(word)
            return np.dot(v, axis) / (np.linalg.norm(v) * np.linalg.norm(axis))

def plot_profession_bias(E, axis, target_file):
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

    target_scores = {w: projection_on_axis(E, w, axis) for w in w2c}
    sorted_targets = sorted(target_scores.items(), key=lambda x: x[1], reverse=True)
    words, scores = zip(*sorted_targets)

    filtered_words = []
    filtered_scores = []
    filtered_colors = []
    filtered_category_names = []

    for w, s in zip(words, scores):
        category, color = w2c[w]
        if color !="#676767":
            filtered_words.append(w)
            filtered_scores.append(s)
            filtered_colors.append(color)
            filtered_category_names.append(category)

    os.makedirs("data/results", exist_ok=True)
    plt.figure(figsize=(20, 6))
    plt.bar(filtered_words, filtered_scores, color=filtered_colors)

    for w in filtered_words:
        plt.axvline(x=w, linestyle='dotted', linewidth=0.7, color='gray')

    maxv = np.max(np.abs(scores))
    plt.ylim(-maxv - 0.1, maxv + 0.1)

    plt.xticks(rotation=90)
    plt.ylabel(f"Axis Cosine Similarity")
    plt.xlabel(f"Profession")
    plt.title(f"Word Embeddings on Axis")

    legend_handles = legent_handles(filtered_category_names, filtered_colors)    
    plt.legend(handles=legend_handles, title="Category", loc='upper right')

    plt.tight_layout()
    plt.savefig(f"data/results/output_professions_gender.png", dpi=300, bbox_inches="tight")
    plt.show()



        

        

