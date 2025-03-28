from flwr_datasets import FederatedDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Adapted from flower tutorial(https://flower.ai/docs/framework/tutorial-series-get-started-with-flower-pytorch.html)
def create_dataloaders_flower(
    num_clients,
    partition_id,
    batch_size,
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