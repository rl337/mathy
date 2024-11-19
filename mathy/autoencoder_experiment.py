import json
import logging
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from mathy.command import Command
from mathy.model.autoencoder import Autoencoder
from mathy.model_trainer import ModelTrainer
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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoded_texts[idx], self.encoded_texts[idx]


# Autoencoder Training Command
class AutoencoderTrainCommand(ModelTrainer):
    dataset: Dataset = None
    tokenizer: CharacterTokenizer = None
    model: nn.Module = None

    def __init__(self):
        super().__init__()

    @classmethod
    def add_args(cls, parser):
        super().add_args(parser)
        parser.add_argument("--corpus-size", type=int, default=100, help="Number of numbers to generate for the corpus")
        parser.add_argument("--max-length", type=int, default=4096, help="Maximum length of the encoded numbers")

    def create_dataset(self) -> Dataset:
        generator = NaturalNumberGenerator(seed=0)
        corpus = generator.generate_batch(self.args.corpus_size)
        corpus_text = [n.rendered for n in corpus]
        logging.info(f"Generated {len(corpus_text)} numbers")

        tokenizer = CharacterTokenizer()
        tokenizer.train(corpus_text)
        logging.info(f"Tokenizer trained with {tokenizer.vocab_size()} tokens")
        return TextDataset(corpus_text, tokenizer, self.args.max_length)
    
    def get_dataset(self) -> Dataset:
        if self.dataset is None:
            self.dataset = self.create_dataset()
        return self.dataset

    def get_criterion(self) -> nn.Module:
        return nn.MSELoss()
    
    def compute_loss(self, criterion: nn.Module, outputs: torch.Tensor, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return criterion(outputs, labels.float())

    def create_model(self) -> nn.Module:
        dataset = self.get_dataset()
        vocab_size = dataset.tokenizer.vocab_size()

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
                
        return Autoencoder(
            vocab_size=vocab_size, 
            embedding_dim=default_config["embedding_dim"], 
            max_length=default_config["max_length"], 
            latent_dim=default_config["latent_dim"],
            encoder_dims=default_config["encoder_dims"],
            decoder_dims=default_config["decoder_dims"],
            output_dim=default_config["max_length"],
        )

    def get_model(self) -> nn.Module:
        if self.model is None:
            self.model = self.create_model()
        return self.model
    
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

