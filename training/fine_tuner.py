from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec 
from gensim.test.utils import datapath
from gensim import utils

import os
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class MyCorpus:
    """Iterates through all .txt files in a folder and preprocess."""

    def __init__(self, folder_path):
        self.folder_path = folder_path

    def __iter__(self):
        for fname in os.listdir(self.folder_path):
            if fname.endswith(".txt"):
                full_path = os.path.join(self.folder_path, fname)
                with open(full_path, "r", encoding="utf8") as f:
                    for line in f:
                        yield utils.simple_preprocess(line)


class EpochLogger(CallbackAny2Vec):
    """Callback to log information about training epochs."""
    def __init__(self):
        self.epoch = 1

    def on_epoch_end(self, model):
        # Log the loss after each epoch
        loss = model.get_latest_training_loss()
        logging.info(f"Finished epoch {self.epoch}. Current total loss: {loss:.4f}")
        self.epoch += 1


def train_word2vec(sentences, vector_size=100, window=5, min_count=2, workers=8, epochs=50):
    
    model = Word2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=1,          
        workers=workers,
        compute_loss=True 
    )

    logging.info("Building vocabulary...")
    logging.info(f"Examples gathered: {model.corpus_count}")
    model.build_vocab(corpus_iterable=sentences)

    logging.info(f"Starting training for {epochs} epochs...")
    
    model.train(
        corpus_iterable=sentences,
        total_examples=model.corpus_count, 
        epochs=epochs,
        callbacks=[EpochLogger()] 
    )
    
    logging.info("Training complete.")

    return model