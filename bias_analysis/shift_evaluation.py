from __future__ import print_function, division
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import numpy as np
import matplotlib.pyplot as plt

def projection_on_axis(E, word, axis):
    v = E.get_vector(word)
    return np.dot(v, axis) / (np.linalg.norm(v) * np.linalg.norm(axis))

def calculate_shift(E, E_new, axis, target_file, bias, target, scale=1):
    with open(target_file, "r") as f:
        # [(w, c)x N]
        target_words = json.load(f)

    target_scores = {w: scale*projection_on_axis(E, w, axis) for w, c in target_words if w in E}
    target_scores_new = {w: scale*projection_on_axis(E_new, w, axis) for w, c in target_words if w in E_new}

    shift = 0
    per_word_shift = {}
    for w in target_scores:
        s1 = target_scores[w]
        s2 = target_scores_new[w]
        d = abs(abs(s1) - abs(s2)) * (1 if abs(s2) < abs(s1) else -1)
        per_word_shift[w] = d

        shift+=d
        # print(w, d)

    words = []
    vals1 = []
    vals2 = []

    for w, c in target_words:
        if w in E and w in E_new:
            v1 = scale * projection_on_axis(E, w, axis)
            v2 = scale * projection_on_axis(E_new, w, axis)
            words.append(w)
            vals1.append(v1)
            vals2.append(v2)

    # plot_shift(per_word_shift, bias, target)
    plot_easy_shift(words, vals1, vals2, bias, target)
    print(f"Shift in {bias} Bias for {target}: ", shift)


def plot_shift(per_word_shift, bias, target):
    # Sort by magnitude (largest change first)
    sorted_items = sorted(per_word_shift.items(), key=lambda x: x[1], reverse=True)
    words, values = zip(*sorted_items)
    sorted_items = sorted(per_word_shift.items(), key=lambda x: x[1], reverse=True)
    words, values = zip(*sorted_items)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.bar(words, values)
    plt.xticks(rotation=90)
    plt.ylabel("Shift")
    plt.title(f"Shift in {bias} Bias for {target}")
    plt.tight_layout()
    plt.show()
    plt.savefig(f"data/results/{bias}_{target}_shifts.png", dpi=300, bbox_inches="tight")


def plot_easy_shift(words, vals1, vals2, bias, target):

    # Filter words that did not flip sign
    filtered_words = []
    filtered_vals1 = []
    filtered_vals2 = []

    for w, v1, v2 in zip(words, vals1, vals2):
        if np.sign(v1) == np.sign(v2):
            filtered_words.append(w)
            filtered_vals1.append(v1)
            filtered_vals2.append(v2)

    filtered = [(w, v1, v2) for w, v1, v2 in zip(filtered_words, filtered_vals1, filtered_vals2) if np.sign(v1) == np.sign(v2)]

    # Sort by original value
    filtered.sort(key=lambda x: x[1])

    # Unpack
    words_sorted, vals1_sorted, vals2_sorted = zip(*filtered) if filtered else ([], [], [])


    # Width offsets
    x = np.arange(len(filtered_words))
    width = 0.4

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, vals1_sorted, width=width, color="orange", alpha=0.6, label="Original")
    plt.bar(x + width/2, vals2_sorted, width=width, color="blue", alpha=0.9, label="Finetuned")

    plt.xticks(x, words_sorted, rotation=90, fontsize=18)
    plt.ylabel(f"Projection on {bias} Bias Axis", fontsize=18)
    plt.xlabel(f"Word (sorted {get_axis_name(bias)})", fontsize=18)
    plt.title(f"Comparison of Original vs Finetuned on {bias} Axis (Same Direction Only)", fontsize=16)

    plt.legend(fontsize=18)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f"data/results/{bias}_{target}_comparison.png", dpi=600, bbox_inches="tight")
    plt.show()

def get_axis_name(bias):
    if bias == "Age":
        return "from Young to Old"
    elif bias == "Gender":
        return "from He to She"
    elif bias == "Class": 
        return "from Poor to Rich"
    
