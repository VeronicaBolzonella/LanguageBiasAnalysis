import argparse
import numpy as np
import torch
from torch.optim import AdamW
from transformers import BertModel


from models.fine_tuner import finetune
from data.data_loader import load_data

def main(epochs=3, batch_size:int=8, folder:str="data/gutenberg_children"):
    model = BertModel.from_pretrained("bert-base-uncased")
    optimizer = AdamW(model.parameters(), lr=2e-5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataloader = load_data(batch_size ,folder)

    embeddings = finetune(dataloader, model, device, optimizer, epochs)

    np.save("models/bert_finetuned_embeddings.npy", embeddings.numpy())
    print("Saved embeddings to bert_finetuned_embeddings.npy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, default="data/gutenberg_children", help="Path to the dataset")
    parser.add_argument("--epochs", type=int, default=3, help="number of training epochs, default to 3")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for finetuning")

    args = parser.parse_args()
    main(epochs=args.epochs, batch_size=args.batch_size, folder=args.data_folder)