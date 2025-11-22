from gensim.models import Word2Vec

def train_word2vec(data, vector_size=100, window=5, min_count=2, workers=8, epochs=50):
    """
    sentences: list of token lists
    returns: trained Word2Vec model
    """

    model = Word2Vec(
        sentences=data,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=1,          
        workers=workers
    )

    model.train(data, total_examples=len(data), epochs=epochs)

    return model


def extract_embeddings(model):
    """
    Returns the word embedding matrix (vocab_size x embedding_size) and 
    the vocabulary list (vocab_size).
    """
    vectors = model.wv.vectors
    vocab = model.wv.index_to_key
    return vectors, vocab
