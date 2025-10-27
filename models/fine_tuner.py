import numpy as np
import torch


def loss_fn(embeddings):
    # TODO: choose loss function
    # Placeholder loss function
    return torch.mean(embeddings.pow(2))


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

