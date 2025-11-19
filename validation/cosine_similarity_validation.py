import os
from sklearn.metrics.pairwise import cosine_similarity

# from https://staff.fnwi.uva.nl/e.bruni/MEN (     Multimodal Distributional Semantics E. Bruni, N. K. Tran and M. Baroni. Journal of Artificial Intelligence Research 49: 1-47. [pdf])




def get_similarity(embeddings, word1, word2, log_file="similarity_log.txt"):
    word1_embedding = embeddings[word1]
    word2_embedding = embeddings[word2]

    sim = cosine_similarity(word1_embedding.reshape(1, -1), word2_embedding.reshape(1, -1))
    message = f"Cosine similarity between {word1} and {word2}: {sim[0][0]:.4f}"

    print(message)

    # Write to log file
    log_path = os.path.join(os.getcwd(), log_file)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(message + "\n")