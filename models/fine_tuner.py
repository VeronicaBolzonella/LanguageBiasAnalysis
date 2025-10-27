import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertModel, BertTokenizer

from data.data_loader import BertEmbeddingDataset, load_books  

def loss_fn(embeddings):
    # TODO: choose loss function
    # Placeholder loss function
    return torch.mean(embeddings.pow(2))


def load_data(batch_size=8):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    sentences = load_books("data/gutenberg_children")
    dataset = BertEmbeddingDataset(sentences, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def finetune(dataloader, model, device, optimizer, epochs=3):
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for batch in dataloader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling

            loss = loss_fn(embeddings)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1} finished.")

    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            all_embeddings.append(embeddings.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_embeddings


def main(epochs=3):
    model = BertModel.from_pretrained("bert-base-uncased")
    optimizer = AdamW(model.parameters(), lr=2e-5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataloader = load_data()

    embeddings = finetune(dataloader, model, device, optimizer, epochs=1)

    np.save("models/bert_finetuned_embeddings.npy", embeddings.numpy())
    print("Saved embeddings to bert_finetuned_embeddings.npy")


if __name__ == "__main__":
    main()
