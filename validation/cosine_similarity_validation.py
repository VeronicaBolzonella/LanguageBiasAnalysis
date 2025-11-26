# from https://staff.fnwi.uva.nl/e.bruni/MEN (     Multimodal Distributional Semantics E. Bruni, N. K. Tran and M. Baroni. Journal of Artificial Intelligence Research 49: 1-47. [pdf])

import pandas as pd
import numpy as np
import io
from scipy.spatial.distance import cosine
from gensim.models import KeyedVectors
from scipy.stats import pearsonr


def calculate_df_similarity(df, embeddings):
    """
    Calculates the cosine similarity between word pairs in a DataFrame
    based on provided embeddings, adds a 'Calculated Score' column,
    and drops rows where words are missing from the embeddings.

    Args:
        df (pd.DataFrame): DataFrame with 'Word 1' and 'Word 2' columns.
        embeddings (dict): Dictionary mapping words (str) to their embedding
                           vectors (np.array).

    Returns:
        pd.DataFrame: The modified DataFrame.
    """
    initial_size = len(df)
    
    df['W1_Exists'] = df['Word 1'].apply(lambda w: w in embeddings)
    df['W2_Exists'] = df['Word 2'].apply(lambda w: w in embeddings)

    df_valid = df[df['W1_Exists'] & df['W2_Exists']].copy()
    
    rows_dropped = initial_size - len(df_valid)
    if rows_dropped > 0:
        print(f"Dropped {rows_dropped} row(s) because one or both words were missing from the embeddings.")

    df_valid.drop(columns=['W1_Exists', 'W2_Exists'], inplace=True)

    def compute_similarity(row):
        emb1 = embeddings.get_vector(row['Word 1'])
        emb2 = embeddings.get_vector(row['Word 2'])

        distance = cosine(emb1, emb2)
        similarity = 1 - distance
        return float(similarity)


    # Apply the calculation across all valid rows
    df_valid['Calculated Score'] = df_valid.apply(compute_similarity, axis=1)

    print(f"Final DataFrame size: {len(df_valid)} rows.")
    return df_valid


def create_similarity_dataframe(file_path):
    """
    Reads a plain text file with 'word1 word2 score' per line,
    and creates a pandas DataFrame.

    Args:
        file_path (str): The path to the plain text file.

    Returns:
        pandas.DataFrame: A DataFrame with columns 'Word 1', 'Word 2', and 'Score'.
    """
    
    word1_list = []
    word2_list = []
    score_list = []

    try:
        # 2. Open and read the file
        with open(file_path, 'r') as f:
            for line in f:
                # Remove leading/trailing whitespace and split the line by spaces
                parts = line.strip().split()
                
                # Check if the line has at least three parts (word1, word2, score)
                if len(parts) >= 3:

                    word1 = parts[0].replace("-n", "")
                    word2 = parts[1].replace("-n", "")

                    score_str = parts[2]
                    
                    # Convert the score to a float, handling potential errors
                    try:
                        # 0:50=0:1
                        score = float(score_str)/50
                    except ValueError:
                        print(f"Warning: Could not convert score '{score_str}' to float on line: {line.strip()}. Skipping line.")
                        continue # Skip to the next line
                        
                    # Append the extracted data to the lists
                    word1_list.append(word1)
                    word2_list.append(word2)
                    score_list.append(score)
                elif line.strip(): # Check if the line is not just empty space
                     print(f"Warning: Skipping malformed line with fewer than 3 parts: {line.strip()}")

    except FileNotFoundError:
        print(f"Error: File not found at path: {file_path}")
        return pd.DataFrame() # Return an empty DataFrame on error

    # 3. Create the DataFrame from the collected lists
    data = {
        'Word 1': word1_list,
        'Word 2': word2_list,
        'Score': score_list
    }
    df = pd.DataFrame(data)
    
    return df


if __name__=="__main__":

    file_to_process = "data/similarity_labelled.dev" 

    similarity_df = create_similarity_dataframe(file_to_process)

    embeddings = KeyedVectors.load("embeddings.kv")
    
    df = calculate_df_similarity(similarity_df, embeddings)

    print(df)
    r, p = pearsonr(df['Score'], df['Calculated Score'])

    print("Pearson correlation:", r)
    print("p-value:", p)

