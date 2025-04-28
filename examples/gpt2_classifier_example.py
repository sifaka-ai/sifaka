import torch
import tiktoken
from transformers import GPT2Model
from torch import nn
from torch.utils.data import Dataset, DataLoader
import random
import json
import os


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=1024):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = 50256  # GPT-2's pad token ID

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize text
        input_ids = self.tokenizer.encode(text)
        # Truncate if needed
        input_ids = input_ids[: min(len(input_ids), self.max_length)]
        # Pad sequence
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [self.pad_token_id] * padding_length

        return torch.tensor(input_ids), torch.tensor(label)


def collate_batch(batch):
    # Separate inputs and labels
    inputs = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    return inputs, labels


class GPT2Classifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Load the pre-trained GPT-2 model
        self.gpt2 = GPT2Model.from_pretrained("gpt2")
        # Add a classification head
        self.classifier = nn.Linear(self.gpt2.config.n_embd, num_classes)

    def forward(self, input_ids):
        # Get GPT-2 outputs
        outputs = self.gpt2(input_ids)
        # Use the last hidden state of the last token for classification
        last_hidden_state = outputs.last_hidden_state[:, -1, :]
        # Pass through the classification head
        logits = self.classifier(last_hidden_state)
        return logits

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=3):
    best_val_acc = 0
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch: {epoch + 1}, Batch: {batch_idx + 1}, Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%"
                )

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_acc = 100.0 * val_correct / val_total
        print(f"\nValidation Accuracy: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save("best_model.pth")

    return train_losses, val_losses


def classify_text(text, model, tokenizer, device, max_length=1024):
    """
    Classify a piece of text using the GPT-2 classifier
    """
    model.eval()

    # Tokenize the input text
    input_ids = tokenizer.encode(text)

    # Truncate if needed
    input_ids = input_ids[: min(len(input_ids), max_length)]

    # Pad sequence
    padding_length = max_length - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + [50256] * padding_length  # GPT-2's pad token ID

    # Convert to tensor and add batch dimension
    input_tensor = torch.tensor([input_ids]).to(device)

    # Get model prediction
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    return predicted_class, confidence


def create_sample_dataset():
    """Create a sample sentiment analysis dataset"""
    dataset = [
        # Positive examples
        ("This product is amazing! I love everything about it.", 1),
        ("Best purchase I've ever made, absolutely wonderful!", 1),
        ("Great value for money, highly recommend!", 1),
        ("Exceeded my expectations, fantastic quality!", 1),
        ("Life-changing product, can't live without it!", 1),
        ("Outstanding customer service and product quality.", 1),
        ("This exceeded all my expectations, truly remarkable!", 1),
        ("Worth every penny, would buy again in a heartbeat.", 1),
        ("The best in its class, nothing else comes close.", 1),
        ("Incredible performance and reliability.", 1),
        ("Perfect for my needs, couldn't be happier!", 1),
        ("Revolutionary product that delivers on all promises.", 1),
        ("Exceptional quality and attention to detail.", 1),
        ("A game-changer in every way possible.", 1),
        ("Absolutely love it, use it every day!", 1),
        # Negative examples
        ("Terrible experience, would not recommend to anyone.", 0),
        ("The quality is decent, but the price is a bit high.", 0),
        ("Not worth the money, very disappointing.", 0),
        ("Average product, nothing special.", 0),
        ("Poor customer service, avoid this company.", 0),
        ("Waste of money, doesn't work as advertised.", 0),
        ("Broke after two weeks, poor build quality.", 0),
        ("Save your money, this is not worth it.", 0),
        ("Disappointing performance, many better alternatives.", 0),
        ("Customer support is non-existent.", 0),
        ("Wouldn't recommend, too many issues.", 0),
        ("Failed to meet basic expectations.", 0),
        ("Overpriced for what you get.", 0),
        ("Frustrating experience from start to finish.", 0),
        ("Don't waste your time with this product.", 0),
        # More nuanced positive examples
        ("Despite minor issues, overall very satisfied.", 1),
        ("Good value for the price point.", 1),
        ("Solid performance, meets all requirements.", 1),
        ("Pleasantly surprised by the quality.", 1),
        ("Does the job well, would recommend.", 1),
        # More nuanced negative examples
        ("Has potential but needs improvement.", 0),
        ("Not terrible, but wouldn't buy again.", 0),
        ("Mediocre performance at best.", 0),
        ("Expected better for the price.", 0),
        ("Too many compromises for the cost.", 0),
    ]

    # Split into train and validation
    random.shuffle(dataset)
    split = int(0.8 * len(dataset))
    train_data = dataset[:split]
    val_data = dataset[split:]

    return train_data, val_data


def main():
    # Set device and random seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    random.seed(42)
    print(f"Using device: {device}")

    # Initialize tokenizer and model
    tokenizer = tiktoken.get_encoding("gpt2")
    model = GPT2Classifier(num_classes=2).to(device)
    print("Model initialized")

    # Create datasets
    train_data, val_data = create_sample_dataset()

    # Create data loaders
    train_dataset = TextClassificationDataset(
        [x[0] for x in train_data], [x[1] for x in train_data], tokenizer, max_length=128
    )
    val_dataset = TextClassificationDataset(
        [x[0] for x in val_data], [x[1] for x in val_data], tokenizer, max_length=128
    )

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=2, collate_fn=collate_batch)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # Train the model
    print("\nStarting training...")
    train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10)

    # Load best model
    if os.path.exists("best_model.pth"):
        model.load("best_model.pth")

    # Test some examples
    print("\nTesting with some examples:")
    test_texts = [
        "This is absolutely fantastic!",
        "I regret buying this.",
        "It's okay, but could be better.",
    ]

    for text in test_texts:
        predicted_class, confidence = classify_text(text, model, tokenizer, device, max_length=128)
        sentiment = "Positive" if predicted_class == 1 else "Negative"
        print(f"\nText: {text}")
        print(f"Prediction: {sentiment}")
        print(f"Confidence: {confidence:.2f}")


if __name__ == "__main__":
    main()
