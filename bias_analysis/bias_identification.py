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
            "S": "STEM",
            "H": "Healthcare",
            "B": "Business & Finance",
            "P": "Government & Law",
            "E": "Education & Social",
            "A": "Arts, Design & Media",
            "F": "Service & Support",
            "C": "Caretaking"
        }
    elif target == 'animals':
        return { 
            "F": "Farm", 
            "P": "Predators", 
            "R": "Reptiles", 
            "S": "Rodents", 
            "I": "Insects", 
            "G": "Grazers", 
            "B": "Birds",
            "A": "Acquatic"
        }
    else:
        return {
            "P": "Positive",
            "N": "Negative",
            "S": "Neutral"
        }

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("embedding_filename", help="The name of the embedding")
    parser.add_argument("bias", help="bias axes between age, gender, class")
    parser.add_argument("target", help="word group between professions, animals, adj")

    args = parser.parse_args()
    category_map = get_category_map(args.target)

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

    # Map words to categories
    word_to_category = {w: category_map[c] for w, c in target_words_list if w in E}

    def projection_on_axis(E, word, axis):
        v = E.get_vector(word)
        return np.dot(v, axis) / (np.linalg.norm(v) * np.linalg.norm(axis))

    target_scores = {w: projection_on_axis(E, w, axis) for w in word_to_category}
    sorted_targets = sorted(target_scores.items(), key=lambda x: x[1], reverse=True)
    words, scores = zip(*sorted_targets)

    # Plot
    categories = sorted(set(word_to_category.values()))
    category_to_idx = {c: i for i, c in enumerate(categories)}
    cmap = plt.get_cmap('tab10')
    colors = [cmap(category_to_idx[word_to_category[w]] / len(categories)) for w in words]

    os.makedirs("data/results", exist_ok=True)
    plt.figure(figsize=(20, 6))
    plt.bar(words, scores, color=colors)
    plt.xticks(rotation=90)
    plt.ylabel(f"Projection on {bias} Axis (Cosine Similarity)")
    plt.xlabel(f"{target}")
    plt.title(f"Word Embeddings Projection on {bias} Axis, Grouped by Category")

    # Legend
    legend_handles = [plt.Rectangle((0, 0), 1, 1, fc=cmap(i / len(categories)), label=cat)
                      for i, cat in enumerate(categories)]
    plt.legend(handles=legend_handles, title="Category", loc='upper right')

    plt.tight_layout()
    plt.savefig(f"data/results/output_{target}_{bias}.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Save results to text
    out_path = "data/results/output.txt"
    with open(out_path, "a") as out:
        out.write(f"{target} projection on {bias} axis\n\n")
        for w, s in sorted_targets:
            out.write(f"{w}: {s:.6f}\n")
        out.write("\n")

