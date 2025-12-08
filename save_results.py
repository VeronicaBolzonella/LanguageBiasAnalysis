from __future__ import print_function, division
import bias_analysis.we as we
import json
import numpy as np
import argparse
from gensim.models import KeyedVectors
import os

def projection_on_axis(E, word, axis):
    v = E.get_vector(word)
    return np.dot(v, axis) / (np.linalg.norm(v) * np.linalg.norm(axis))
def combined_scores_for_target(E, E_new, axis_dict, target_file):
    # axis_dict = {"Gender": gender_axis, "Age": age_axis, "Class": class_axis}
    with open(target_file, "r") as f:
        target_words = json.load(f)

    # Extract words only (ignore category fields)
    if len(target_words[0]) == 2:
        words = [w for w, c in target_words]
    else:
        words = [w for w, c, c2 in target_words]

    # Compute all projections
    result = {}
    for w in words:
        if w not in E or w not in E_new:
            continue

        result[w] = {}
        for bias_name, axis in axis_dict.items():
            scale = -1 if bias_name == "Class" else 1

            orig_val = scale * projection_on_axis(E, w, axis)
            new_val  = scale * projection_on_axis(E_new, w, axis)

            result[w][f"{bias_name}_orig"] = float(orig_val)
            result[w][f"{bias_name}_new"]  = float(new_val)

    # Convert dict → list
    out = []
    for w, vals in result.items():
        entry = {"word": w}
        entry.update(vals)
        out.append(entry)

    return out


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("embedding_filename")
    parser.add_argument("embedding_new_filename")
    args = parser.parse_args()

    # Load equalization pairs
    age_direction_file = "data/terms/age_equalize_pairs.json"
    gender_direction_file = "data/terms/gender_equalize_pairs.json"
    class_direction_file = "data/terms/class_equalize_pairs.json"

    adj_target_file = "data/terms/sadjectives.json"
    prof_target_file = "data/terms/sprofessions.json"
    animals_target_file = "data/terms/sanimals.json"

    with open(age_direction_file) as f: age_defs = json.load(f)
    with open(gender_direction_file) as f: gender_defs = json.load(f)
    with open(class_direction_file) as f: class_defs = json.load(f)

    E = KeyedVectors.load(args.embedding_filename)
    E_new = KeyedVectors.load(args.embedding_new_filename)

    age_axis = we.doPCA(age_defs, E).components_[0]
    gender_axis = we.doPCA(gender_defs, E).components_[0]
    class_axis = we.doPCA(class_defs, E).components_[0]

    axis_dict = {
        "Gender": gender_axis,
        "Age": age_axis,
        "Class": class_axis
    }

    # Process three target groups
    animals_out = combined_scores_for_target(E, E_new, axis_dict, animals_target_file)
    professions_out = combined_scores_for_target(E, E_new, axis_dict, prof_target_file)
    adjectives_out = combined_scores_for_target(E, E_new, axis_dict, adj_target_file)

    os.makedirs("data/results", exist_ok=True)

    with open("data/results/animals_all_biases.json", "w") as f:
        json.dump(animals_out, f, indent=2)

    with open("data/results/professions_all_biases.json", "w") as f:
        json.dump(professions_out, f, indent=2)

    with open("data/results/adjectives_all_biases.json", "w") as f:
        json.dump(adjectives_out, f, indent=2)
