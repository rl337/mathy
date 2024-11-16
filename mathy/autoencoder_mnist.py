import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List
from mathy.command import Command
from mathy.model.autoencoder import ClassicAutoencoder
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
class AutoencoderMNistCommand(Command):
    def __init__(self):
        super().__init__()

    @classmethod
    def add_args(cls, parser):
        pass

    def action(self):
        model = ClassicAutoencoder(latent_dim=48)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0005)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, threshold=0.01, min_lr=1e-6
        )

        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1] range (mean=0.5, std=0.5)
        ])

        # Download and load the training dataset
        train_dataset = datasets.MNIST(root='mnist_data', train=True, download=True, transform=transform)

        # split into train and validation
        val_size = 20
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

        # Create the DataLoader for batching
        train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
        # Training loop
        epochs = 10
        epoch = 0
        last_epoch_loss = float('inf')
        while epoch < epochs:
            model.train()

            orig_model_state_dict = model.state_dict()

            total_loss = 0.0
            for images, _ in train_loader:  
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, images)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            learning_rate = optimizer.param_groups[0]['lr']
            logging.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Learning Rate: {learning_rate:.6f}")
            scheduler.step(avg_loss)

            model.eval()
            epoch_validation_dataset = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)
            with torch.no_grad():
                for images, _ in epoch_validation_dataset:
                    outputs = model(images)
                    _ = criterion(outputs, images)

                    image_pairs = list(zip(images, outputs))    
                    side_by_side_image = create_side_by_side_image(image_pairs)
                    side_by_side_image.save(f"epoch_{epoch+1}_validation.png")

            if last_epoch_loss - avg_loss > 0.005:
                last_epoch_loss = avg_loss
                epoch += 1
            else:
                logging.info(f"Loss did not improve, not counting this epoch")
                model.load_state_dict(orig_model_state_dict)
