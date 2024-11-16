import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List
from mathy.command import Command
from mathy.model.autoencoder import ClassicAutoencoder
from mathy.model_trainer import EpochContext, ModelTrainer
from mathy.tokenizer import CharacterTokenizer
from torchvision import datasets, transforms
from PIL import Image

    
# Function to create a single large image displaying all pairs
def create_side_by_side_image(image_pairs):
    # Determine the dimensions based on the first image
    original_image, _ = image_pairs[0]
    single_width, single_height = transforms.ToPILImage()(original_image.squeeze()).size
    total_width = single_width * 2
    total_height = single_height * len(image_pairs)
    
    # Create a new blank image with the computed size
    final_image = Image.new("L", (total_width, total_height))
    
    # Paste each (original, reconstructed) pair into the final image
    for i, (original, reconstructed) in enumerate(image_pairs):
        # Convert tensors to Pillow images
        original_image = transforms.ToPILImage()(original.squeeze())
        reconstructed_image = transforms.ToPILImage()(reconstructed.squeeze())
        
        # Paste the original and reconstructed side by side
        final_image.paste(original_image, (0, i * single_height))
        final_image.paste(reconstructed_image, (single_width, i * single_height))
    
    return final_image


# Autoencoder Training Command
class AutoencoderMNistCommand(ModelTrainer):
    def __init__(self):
        super().__init__()

    def get_model(self) -> nn.Module:
        return ClassicAutoencoder(latent_dim=48)

    def get_optimizer(self, model: nn.Module, learning_rate: float) -> optim.Optimizer:
        return optim.Adam(model.parameters(), lr=learning_rate)

    def get_criterion(self) -> nn.Module:
        return nn.MSELoss()

    def get_dataset(self) -> Dataset:
        return datasets.MNIST(root='mnist_data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1] range (mean=0.5, std=0.5)
        ]))
    
    def compute_loss(self, criterion: nn.Module, outputs: torch.Tensor, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return criterion(outputs, inputs)
    
    def post_validation_action(self, epoch_context: EpochContext):
        super().post_validation_action(epoch_context)

        # go through first 10 images in validation set run the model on them as a batch and zip the results with the original images
        # then save the results as a side by side image
        
        # loop through first 10 images in validation set
        image_pairs = []

        for i in range(10):
            input, _ = epoch_context.validation_dataset[i]
            output, _ = epoch_context.model(input.to(epoch_context.device))
            image_pairs.append((input, output))

        side_by_side_image = create_side_by_side_image(image_pairs)
        side_by_side_image.save(f"epoch_{epoch_context.epoch}_validation.png")

