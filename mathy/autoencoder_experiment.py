import json
import logging
import random
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

    def __init__(self, corpus: List[str], tokenizer: CharacterTokenizer, context_length: int):
        self.texts = corpus
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
            "embedding_dim": 32,
            "max_length": 4096,
            "latent_dim": 32,
            "epochs": 10,
            "batch_size": 32,
            "learning_rate": 0.0001,
            "encoder_dims": [512, 256],
            "decoder_dims": [256, 512],
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
        encoder_dims = config["encoder_dims"]
        decoder_dims = config["decoder_dims"]

        corpus_size = 10000
        validation_set_percentage = 0.2

        # TODO: add corpus and validation set file support
        generator = NaturalNumberGenerator(seed=0)

        corpus = generator.generate_batch(corpus_size)
        corpus_text = [n.rendered for n in corpus]
        logging.info(f"Generated {len(corpus_text)} numbers")

        training_set = corpus_text[:int(corpus_size * (1 - validation_set_percentage))]
        validation_set = corpus_text[int(corpus_size * (1 - validation_set_percentage)):]

        tokenizer = CharacterTokenizer()
        tokenizer.train(corpus_text)
        logging.info(f"Tokenizer trained with {tokenizer.vocab_size()} tokens")

        # Initialize datasets and dataloaders
        train_dataset = TextDataset(training_set, tokenizer, max_length)
        val_dataset = TextDataset(validation_set, tokenizer, max_length)
        logging.info(f"Created training and validation datasets with {len(train_dataset)} and {len(val_dataset)} examples")
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # choose first 10 samples from the validation set
        samples = val_dataset[:10]
        samples_decoded_original = [tokenizer.decode(t.tolist()) for t in samples]
        logging.info(f"Sampled 10 numbers from validation set: {samples_decoded_original}")

        vocab_size = tokenizer.vocab_size()
        cuda_available = torch.cuda.is_available()
        device = torch.device("cuda" if cuda_available else "cpu")
        logging.info(f"Using {'GPU' if cuda_available else 'CPU'} for training")

        # Initialize model, criterion, and optimizer
        model = Autoencoder(
            vocab_size=vocab_size, 
            embedding_dim=embedding_dim, 
            max_length=max_length, 
            latent_dim=latent_dim,
            encoder_dims=encoder_dims,
            decoder_dims=decoder_dims,
            output_dim=max_length
        ).to(device)
        criterion = nn.CrossEntropyLoss(reduction="mean")
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training and validation loop
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            train_loss = self.train_autoencoder(model, train_loader, criterion, optimizer, device)
            val_loss = self.validate_autoencoder(model, val_loader, criterion, device)
            print(f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

            # sample 10 numbers from the validation set and decode them
            decoded_samples = self.decode_samples(model, samples, tokenizer, device)
            for i, (original, decoded) in enumerate(zip(samples_decoded_original, decoded_samples)):
                logging.info(f"Sample {i+1} - Original: {original}, Decoded: {decoded}")

    def output_to_token_ids(self, tokenizer: CharacterTokenizer, outputs: torch.Tensor) -> List[int]:
        rounded = torch.round(outputs)
        clamped = torch.clamp(rounded, min=0, max=tokenizer.vocab_size() - 1)
        return clamped.tolist()

    def decode_samples(self, model, samples, tokenizer, device):
        model.eval()
        with torch.no_grad():
            inputs = torch.stack(samples).to(device)
            outputs, _ = model(inputs)
            token_ids = self.output_to_token_ids(tokenizer, outputs)
            decoded_samples = [tokenizer.decode(token_ids[i]) for i in range(len(token_ids))]
        return decoded_samples

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
