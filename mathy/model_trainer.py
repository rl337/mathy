from argparse import Namespace
import logging
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from mathy.command import Command

from mathy.command import not_a_command

class EpochContext:
    epoch: int
    model: nn.Module
    training_dataset: Dataset
    validation_dataset: Dataset
    device: str

    def __init__(self, epoch: int, model: nn.Module, training_dataset: Dataset, validation_dataset: Dataset, device: str):
        self.epoch = epoch
        self.model = model
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.device = device


@not_a_command
class ModelTrainer(Command):

    def __init__(self):
        super().__init__()

    @classmethod
    def add_args(cls, parser):
        parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train for")
        parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate for the optimizer")
        parser.add_argument("--validation-set-percentage", type=float, default=0.2, help="Percentage of the dataset to use as a validation set")
        parser.add_argument("--validation-set-count", type=Optional[int], default=None, help="Number of samples to use as a validation set. Overrides validation-set-percentage")
        parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training and validation")

    def initialize(self, args: Namespace):
        super().initialize(args)

    def get_model(self) -> nn.Module:
        raise NotImplementedError("Subclasses must implement the get_model method")

    def get_optimizer(self, model: nn.Module, learning_rate: float) -> optim.Optimizer:
        return optim.Adam(model.parameters(), lr=learning_rate)

    def get_criterion(self) -> nn.Module:
        raise NotImplementedError("Subclasses must implement the get_criterion method")

    def get_scheduler(self, optimizer: optim.Optimizer) -> optim.lr_scheduler._LRScheduler:
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, threshold=0.01, min_lr=1e-6
        )
    
    def get_dataset(self) -> Dataset:
        raise NotImplementedError("Subclasses must implement the get_dataset method")
    
    def split_dataset(self, dataset: Dataset, validation_set_percentage: float = 0.2, validation_set_count: Optional[int] = None) -> Tuple[Dataset, Dataset]:
        if validation_set_count is None:
            validation_set_count = int(len(dataset) * validation_set_percentage)
        return torch.utils.data.random_split(dataset, [len(dataset) - validation_set_count, validation_set_count])


    def compute_loss(self, criterion: nn.Module, outputs: torch.Tensor, inputs: torch.Tensor, labels: torch.Tensor) -> float:
        # by default compute loss between outputs and labels
        return criterion(outputs, labels)
    def get_transform(self) -> transforms.Compose:
        return None
    
    def get_device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    def pre_train_action(self, epoch_context: EpochContext):
        epoch_context.model.train()

    def post_train_action(self, epoch_context: EpochContext):
        pass

    def pre_validation_action(self, epoch_context: EpochContext):
        epoch_context.model.eval()

    def post_validation_action(self, epoch_context: EpochContext):
        pass

    def action(self):

        device = self.get_device()

        model = self.get_model()
        model.to(device)

        optimizer = self.get_optimizer(model, self.args.learning_rate)
        criterion = self.get_criterion()
        scheduler = self.get_scheduler(optimizer)
        
        dataset = self.get_dataset()
        train_dataset, val_dataset = self.split_dataset(dataset)

        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=True)
        
        epoch = 0
        while epoch < self.args.epochs:
            epoch += 1

            epoch_context = EpochContext(epoch, model, train_dataset, val_dataset, device)
            self.pre_train_action(epoch_context)

            total_train_loss = 0.0
            for inputs, labels in train_loader:  
                in_device_inputs = inputs.to(device)
                in_device_labels = labels.to(device)
                optimizer.zero_grad()
                outputs, _ = model(in_device_inputs)
                loss = self.compute_loss(criterion, outputs, in_device_inputs, in_device_labels)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            avg_train_loss = total_train_loss / len(train_loader)
            learning_rate = optimizer.param_groups[0]['lr']    
            scheduler.step(avg_train_loss)
            self.post_train_action(epoch_context)

            self.pre_validation_action(epoch_context)
            total_validation_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    in_device_inputs = inputs.to(device)
                    outputs, _ = model(in_device_inputs)
                    loss = self.compute_loss(criterion, outputs, in_device_inputs, in_device_labels)
                    total_validation_loss += loss.item()
            avg_validation_loss = total_validation_loss / len(val_loader)
            self.post_validation_action(epoch_context)

            logging.info(f"Epoch [{epoch}/{self.args.epochs}], Train Loss: {avg_train_loss:.4f}, Learning Rate: {learning_rate:.6f}, Val Loss: {avg_validation_loss:.4f}")
 
            
