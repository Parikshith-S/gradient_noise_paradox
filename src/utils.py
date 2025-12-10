import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os


def get_dataloaders(batch_size, data_root="./data"):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_set = datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform
    )
    test_set = datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def plot_results(history, save_path="./results/tradeoff_plot.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Accuracy (%)", color="tab:blue")
    ax1.plot(history["clean"], label="Clean Acc", color="tab:blue", linestyle="--")
    ax1.plot(history["robust"], label="Robust Acc", color="tab:blue", linewidth=2)
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Privacy Cost (ε)", color="tab:red")
    ax2.plot(history["epsilon"], label="Privacy Budget (ε)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.legend(loc="upper right")

    plt.title("The Gradient Noise Paradox: Robustness vs Privacy")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
