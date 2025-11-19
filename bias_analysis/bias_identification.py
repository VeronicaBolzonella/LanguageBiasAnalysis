from __future__ import print_function, division
import we
import json
import numpy as np
import argparse
import sys
if sys.version_info[0] < 3:
    import io
    open = io.open

"""
Hard-debias embedding

Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings
Tolga Bolukbasi, Kai-Wei Chang, James Zou, Venkatesh Saligrama, and Adam Kalai
2016
"""

def get_direction_axis(E, gender_specific_words, definitional, equalize):
    return we.doPCA(definitional, E).components_[0]
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("embedding_filename", help="The name of the embedding")
    parser.add_argument("definitional_filename", help="JSON of definitional pairs")
    parser.add_argument("gendered_words_filename", help="File containing words not to neutralize (one per line)")
    parser.add_argument("equalize_filename", help="???.bin")
    parser.add_argument("debiased_filename", help="???.bin")

    args = parser.parse_args()
    print(args)

    with open(args.definitional_filename, "r") as f:
        defs = json.load(f)
    print("definitional", defs)

    with open(args.equalize_filename, "r") as f:
        equalize_pairs = json.load(f)

    with open(args.gendered_words_filename, "r") as f:
        gender_specific_words = json.load(f)
    print("gender specific", len(gender_specific_words), gender_specific_words[:10])

    E = we.WordEmbedding(args.embedding_filename)

    get_direction_axis(E, gender_specific_words, defs, equalize_pairs)

    