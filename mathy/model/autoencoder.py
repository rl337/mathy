import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, List, Optional, Tuple

class ClassicAutoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super(ClassicAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Flatten(),                      
            nn.Linear(784, 128),                
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)           
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 784),                 
            nn.Sigmoid(),                        
            nn.Unflatten(1, (1, 28, 28))        
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class Autoencoder(nn.Module):
    activation: Callable[[torch.Tensor], torch.Tensor]
    output_activation: Optional[Callable[[torch.Tensor], torch.Tensor]]
    embedding: nn.Embedding
    encoder_layers: nn.ModuleList
    decoder_layers: nn.ModuleList

    def __init__(self, 
                 vocab_size: int, 
                 embedding_dim: int, 
                 max_length: int, 
                 latent_dim: int,
                 encoder_dims: List[int] = [128, 64], 
                 decoder_dims: List[int] = [64, 128], 
                 output_dim: int = 1, 
                 activation: Callable[[torch.Tensor], torch.Tensor] = F.relu, 
                 output_activation: Optional[Callable[[torch.Tensor], torch.Tensor]] = None) -> None:
        """
        Initialize the autoencoder model.

        Parameters:
        - vocab_size (int): The size of the vocabulary for embedding.
        - embedding_dim (int): The dimension of each embedding vector.
        - max_length (int): The maximum length of input sequences.
        - latent_dim (int): The dimension of the latent space representation.
        - encoder_dims (List[int]): Dimensions for each encoder layer.
        - decoder_dims (List[int]): Dimensions for each decoder layer.
        - output_dim (int): Dimension of the final output layer.
        - activation (Callable): Activation function for hidden layers.
        - output_activation (Optional[Callable]): Activation function for output layer, if any.
        """
        super(Autoencoder, self).__init__()
        
        # Assign activation functions
        self.activation = activation
        self.output_activation = output_activation

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        input_dim = embedding_dim * max_length
        for dim in encoder_dims:
            self.encoder_layers.append(nn.Linear(input_dim, dim))
            input_dim = dim
        self.encoder_layers.append(nn.Linear(input_dim, latent_dim))  # Latent space representation

        # Decoder layers
        self.decoder_layers = nn.ModuleList()
        input_dim = latent_dim
        for dim in decoder_dims:
            self.decoder_layers.append(nn.Linear(input_dim, dim))
            input_dim = dim
        self.decoder_layers.append(nn.Linear(input_dim, output_dim))  # Final output layer

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the autoencoder.

        Parameters:
        - x (torch.Tensor): Input tensor of token indices.

        Returns:
        - Tuple[torch.Tensor, torch.Tensor]: Output tensor and latent representation.
        """
        # Embedding
        x = self.embedding(x)
        x = x.view(x.size(0), -1)  # Flatten embeddings into a vector

        # Encoder
        for layer in self.encoder_layers[:-1]:  # All except last encoder layer
            x = self.activation(layer(x))
        latent = self.encoder_layers[-1](x)  # Latent space

        # Decoder
        x = latent
        for layer in self.decoder_layers[:-1]:  # All except last decoder layer
            x = self.activation(layer(x))
        x = self.decoder_layers[-1](x)  # Final output layer
        if self.output_activation:
            x = self.output_activation(x)  # Apply output activation if specified

        return x, latent
