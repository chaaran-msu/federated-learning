from typing import Tuple, List
from flwr_datasets import FederatedDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset

# Adapted from flower tutorial(https://flower.ai/docs/framework/tutorial-series-get-started-with-flower-pytorch.html)
def create_dataloaders_flower(
    num_clients: int,
    partition_id: int,
    batch_size: int,
):
    # Initialize the dataset
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": num_clients})\
    
    # Load dataset for this device
    partition = fds.load_partition(partition_id)

    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    # Transformations
    pytorch_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        # Instead of passing transforms to CIFAR10(..., transform=transform)
        # we will use this function to dataset.with_transform(apply_transforms)
        # The transforms object is exactly the same
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    # Create train/val for each partition and wrap it into DataLoader
    partition_train_test = partition_train_test.with_transform(apply_transforms)

    # Create the train/val loaders
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True
    )
    valloader = DataLoader(partition_train_test["test"], batch_size=batch_size)

    return trainloader, valloader

def create_dataloaders_flower_multiple_partitions(
    num_clients: int,
    partition_ids: List[str],
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    # Initialize the federated dataset
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": num_clients}, seed=42)

    # Transformations
    pytorch_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    # Lists to collect train/test datasets from all specified partitions
    train_datasets = []
    val_datasets = []

    for pid in partition_ids:
        partition = fds.load_partition(pid)
        partition = partition.train_test_split(test_size=0.2, seed=42)
        partition = partition.with_transform(apply_transforms)
        train_datasets.append(partition["train"])
        val_datasets.append(partition["test"])

    # Combine the datasets
    combined_train_dataset = ConcatDataset(train_datasets)
    combined_val_dataset = ConcatDataset(val_datasets)

    # Create DataLoaders
    trainloader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(combined_val_dataset, batch_size=batch_size)

    return trainloader, valloader

def create_global_test_dataloader(
    num_clients: int,
    batch_size: int
):
    # Initialize FederatedDataset (partitioning train across clients and one global test set)
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": num_clients, "test": 1}, seed=42)

    # Load the global test set using the partition 0 of split "test"
    global_test_dataset = fds.load_partition(0, "test")

    # Define transforms
    pytorch_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Apply transforms
    def apply_transforms(batch):
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    global_test_dataset = global_test_dataset.with_transform(apply_transforms)

    # Wrap in DataLoader
    test_loader = DataLoader(global_test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader