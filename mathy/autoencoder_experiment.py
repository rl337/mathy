import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List
from mathy.command import Command
from mathy.model.autoencoder import Autoencoder
from mathy.tokenizer import CharacterTokenizer
from mathy.synthetic.numbers import NaturalNumberGenerator, Number

# Text Dataset
class TextDataset(Dataset):
    texts: List[str]
    tokenizer: CharacterTokenizer
    context_length: int
    encoded_texts: List[torch.Tensor]

    def __init__(self, corpus: List[Number], tokenizer: CharacterTokenizer, context_length: int):
        self.texts = [number.spelled_out for number in corpus]
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.encoded_texts = [self.encode_text(text) for text in self.texts]

    def encode_text(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer.tokenize(text, context_length=self.context_length)
        return torch.tensor(tokens, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.encoded_texts)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.encoded_texts[idx]

# Autoencoder Training Command
class AutoencoderTrainCommand(Command):
    def __init__(self):
        super().__init__()

    @classmethod
    def add_args(cls, parser):
        pass

    def action(self):

        default_config = {
            "embedding_dim": 16,
            "max_length": 100,
            "latent_dim": 32,
            "epochs": 10,
            "batch_size": 32,
            "learning_rate": 0.001
        }

        # start with default config but if a config file is provided, override the defaults
        config = default_config

        # TODO: add config file support

        embedding_dim = config["embedding_dim"]
        max_length = config["max_length"]
        latent_dim = config["latent_dim"]
        epochs = config["epochs"]
        batch_size = config["batch_size"]
        learning_rate = config["learning_rate"]

        corpus_size = 10000
        validation_set_percentage = 0.1

        # TODO: add corpus and validation set file support
        generator = NaturalNumberGenerator(seed=0)

        corpus = generator.generate_batch(corpus_size)
        training_set = corpus[:int(corpus_size * (1 - validation_set_percentage))]
        validation_set = corpus[int(corpus_size * (1 - validation_set_percentage)):]

        tokenizer = CharacterTokenizer()
        tokenizer.train([n.spelled_out for n in corpus])

        # Initialize datasets and dataloaders
        train_dataset = TextDataset(training_set, tokenizer, max_length)
        val_dataset = TextDataset(validation_set, tokenizer, max_length)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        vocab_size = tokenizer.vocab_size()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model, criterion, and optimizer
        model = Autoencoder(vocab_size=vocab_size, embedding_dim=embedding_dim, max_length=max_length, latent_dim=latent_dim).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training and validation loop
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            train_loss = self.train_autoencoder(model, train_loader, criterion, optimizer, device)
            val_loss = self.validate_autoencoder(model, val_loader, criterion, device)
            print(f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    def train_autoencoder(self, model, dataloader, criterion, optimizer, device):
        model.train()
        running_loss = 0.0
        for inputs in dataloader:
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, inputs.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        return running_loss / len(dataloader)

    def validate_autoencoder(self, model, dataloader, criterion, device):
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for inputs in dataloader:
                inputs = inputs.to(device)
                outputs, _ = model(inputs)
                loss = criterion(outputs, inputs.float())
                running_loss += loss.item()
        return running_loss / len(dataloader)
