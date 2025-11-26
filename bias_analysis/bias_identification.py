from __future__ import print_function, division
import we
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from gensim.models import KeyedVectors
import os


def get_direction_axis(E, definitional):
    return we.doPCA(definitional, E).components_[0]

def get_category_map(target):
    if target == 'professions':
        return {
            "S": ("STEM", "#1f77b4"),
            "H": ("Doctors", "#ff7f0e"),
            "B": ("Business & Finance", "#2ca02c"),
            "P": ("Government & Law", "#d62728"),
            "E": ("Education and Academia", "#9467bd"),
            "A": ("Arts, Design & Media", "#8c564b"),
            "F": ("Service & Support", "#e377c2"),
            "C": ("Caregiving", "#7f7f7f")
        }

    elif target == 'animals':
        return { 
            "P": ("Predator", "#d62728"),
            "N": ("Neutral", "#1f77b4"),
            "V": ("Prey", "#2ca02c")
        }

    else:
        return {
            "P": ("Positive", "#2ca02c"),
            "N": ("Negative", "#d62728"),
            "S": ("Neutral", "#676767")
        }

    

def get_category_map2(target):
    if target == 'professions':
        return {
            
        }
    elif target == 'animals':
        return { 
            "B": ("Big", "#d62728"),
            "S": ("Small", "#2ca02c"),
            "M": ("Medium",  "#676767")
        }
    else:
        return {
            
        }
    

def plot(words, scores, category_keys, target, bias, i):
    """
    words: list of words
    scores: corresponding projection scores
    category_keys: category letters, e.g., "P", "N", "S"
    """
    if i == 0:
        category_map = get_category_map(target)
    else:
        category_map = get_category_map2(target)    

    filtered_words = []
    filtered_scores = []
    filtered_colors = []
    filtered_category_names = []

    for w, s, c_key in zip(words, scores, category_keys):
        if c_key not in category_map:
            continue
        cat_name, color = category_map[c_key]

        if color.lower() == "#676767":  # skip gray
            continue

        filtered_words.append(w)
        filtered_scores.append(s)
        filtered_colors.append(color)
        filtered_category_names.append(cat_name)

    # Build legend from remaining categories
    unique_legend = {}
    for name, col in zip(filtered_category_names, filtered_colors):
        unique_legend[name] = col

    os.makedirs("data/results", exist_ok=True)
    plt.figure(figsize=(20, 6))
    plt.bar(filtered_words, filtered_scores, color=filtered_colors)

    for w in filtered_words:
        plt.axvline(x=w, linestyle='dotted', linewidth=0.7, color='gray')

    maxv = np.max(np.abs(scores))
    plt.ylim(-maxv - 0.1, maxv + 0.1)

    plt.xticks(rotation=90)
    plt.ylabel(f"Axis Cosine Similarity {bias}")
    plt.xlabel(f"{target}")
    plt.title(f"Word Embeddings on {bias} Axis")

    # Legend
    legend_handles = [plt.Rectangle((0, 0), 1, 1, fc=col, label=name)
                      for name, col in unique_legend.items()]
    plt.legend(handles=legend_handles, title="Category", loc='upper right')

    plt.tight_layout()
    plt.savefig(f"data/results/output_{target}_{bias}_{i}.png", dpi=300, bbox_inches="tight")
    plt.show()

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
    axis = get_direction_axis(E, defs)

    with open(target_file, "r") as f:
        target_words_list = json.load(f)

        def projection_on_axis(E, word, axis):
            v = E.get_vector(word)
            return np.dot(v, axis) / (np.linalg.norm(v) * np.linalg.norm(axis))

        
    # Map words to categories
    category_map = get_category_map(args.target)

    if len(target_words_list[0]) == 3:
        word_to_category = {w: category_map[c] for w, c, _ in target_words_list if w in E}
    else:
        word_to_category = {w: category_map[c] for w, c in target_words_list if w in E}

    target_scores = {w: projection_on_axis(E, w, axis) for w in word_to_category}
    sorted_targets = sorted(target_scores.items(), key=lambda x: x[1], reverse=True)
    words, scores = zip(*sorted_targets)

    # Plot
    categories = sorted(set(word_to_category.values()))

    # category_keys = just the letter for each word
    if len(target_words_list[0]) == 3:
        category_keys = [c for w, c, _ in target_words_list if w in E]
    else:
        category_keys = [c for w, c in target_words_list if w in E]


    print("Plotting first plot")
    plot(words, scores, category_keys, target, bias, 0)
    print("Done")

    if len(target_words_list[0]) == 3:
        category_map2 = get_category_map2(args.target)
        word_to_category2 = {w: category_map2[c2] for w, c1, c2 in target_words_list if w in E}

        target_scores = {w: projection_on_axis(E, w, axis) for w in word_to_category2}
        sorted_targets = sorted(target_scores.items(), key=lambda x: x[1], reverse=True)
        words, scores = zip(*sorted_targets)

        # Plot
        categories = sorted(set(word_to_category2.values()))

        # category_keys = just the letter for each word
        words, scores = zip(*sorted_targets)
        category_keys = [next(c2 for w2, c1, c2 in target_words_list if w2 == w) for w in words]
        print("Plotting second plot")
        plot(words, scores, category_keys, target, bias, 1)


    # Save results to text
    out_path = "data/results/output.txt"
    with open(out_path, "a") as out:
        out.write(f"{target} projection on {bias} axis\n\n")
        for w, s in sorted_targets:
            out.write(f"{w}: {s:.6f}\n")
        out.write("\n")

