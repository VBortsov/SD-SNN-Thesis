import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from generators.basic.basicSignalGenerator import basicSignalGenerator
from decomposers.ML_methods.NN_based.models.unet1d.unet1d import UNet1D


class SyntheticDecompositionDataset(Dataset):
    def __init__(self, num_samples: int, signal_length: int = 1024, fs: int = 256):
        self.num_samples = num_samples
        self.signal_length = signal_length
        self.fs = fs
        self.duration = signal_length / fs
        self.t = np.linspace(0, self.duration, signal_length, endpoint=False)

    def __len__(self):
        return self.num_samples

    def _make_harmonic(self):
        n_harm = random.randint(1, 4)
        amplitudes = np.random.uniform(0.2, 1.0, size=n_harm)
        frequencies = np.random.uniform(2.0, 40.0, size=n_harm)
        phases = np.random.uniform(0.0, 2 * np.pi, size=n_harm)

        return basicSignalGenerator.harmonic_mixture_generator(
            self.t,
            amplitudes=amplitudes,
            frequencies=frequencies,
            phases=phases
        )

    def _make_amfm(self):
        a0 = np.random.uniform(0.5, 1.2)
        a1 = np.random.uniform(0.1, 0.5)
        fa = np.random.uniform(0.1, 1.0)

        f0 = np.random.uniform(5.0, 20.0)
        f1 = np.random.uniform(1.0, 8.0)
        ff = np.random.uniform(0.05, 0.5)

        return basicSignalGenerator.am_fm_mode_generator(
            self.t,
            amplitude_envelope=lambda tt: a0 + a1 * np.sin(2 * np.pi * fa * tt),
            instantaneous_frequency=lambda tt: f0 + f1 * np.sin(2 * np.pi * ff * tt),
            phase0=np.random.uniform(0.0, 2 * np.pi)
        )

    def _make_chirp(self):
        order = random.choice([1, 2, 3])

        c0 = np.random.uniform(3.0, 15.0)
        coeffs = [c0]

        if order >= 1:
            coeffs.append(np.random.uniform(2.0, 25.0))
        if order >= 2:
            coeffs.append(np.random.uniform(-8.0, 8.0))
        if order >= 3:
            coeffs.append(np.random.uniform(-3.0, 3.0))

        return basicSignalGenerator.chirp_signal_generator(
            self.t,
            coefficients=coeffs,
            amplitude=np.random.uniform(0.5, 1.0),
            phase0=np.random.uniform(0.0, 2 * np.pi)
        )

    @staticmethod
    def _normalize(x, eps=1e-8):
        return x / (np.max(np.abs(x)) + eps)

    def __getitem__(self, idx):
        harmonic = self._make_harmonic()
        amfm = self._make_amfm()
        chirp = self._make_chirp()

        harmonic = self._normalize(harmonic)
        amfm = self._normalize(amfm)
        chirp = self._normalize(chirp)

        components = np.stack([harmonic, amfm, chirp], axis=0)
        mixture = np.sum(components, axis=0)

        mixture = self._normalize(mixture)

        noise_std = np.random.uniform(0.0, 0.05)
        mixture = mixture + np.random.normal(0.0, noise_std, size=mixture.shape)

        mixture = mixture.astype(np.float32)[None, :]
        components = components.astype(np.float32)

        return torch.from_numpy(mixture), torch.from_numpy(components)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        y_hat = model(x)
        loss = criterion(y_hat, y)
        running_loss += loss.item() * x.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def plot_example(model, dataset, device, index=0):
    model.eval()

    x, y = dataset[index]
    x_in = x.unsqueeze(0).to(device)

    y_hat = model(x_in).cpu().squeeze(0).numpy()
    x = x.squeeze(0).numpy()
    y = y.numpy()

    t = dataset.t

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(t, x)
    axes[0].set_title("Mixture")
    axes[0].grid(True)

    names = ["Harmonic", "AM-FM", "Chirp"]
    for i in range(3):
        axes[i + 1].plot(t, y[i], label=f"True {names[i]}")
        axes[i + 1].plot(t, y_hat[i], "--", label=f"Pred {names[i]}")
        axes[i + 1].set_title(names[i])
        axes[i + 1].legend()
        axes[i + 1].grid(True)

    axes[-1].set_xlabel("Time [s]")
    plt.tight_layout()
    plt.show()


def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = SyntheticDecompositionDataset(num_samples=2000, signal_length=1024, fs=256)
    val_dataset = SyntheticDecompositionDataset(num_samples=300, signal_length=1024, fs=256)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    model = UNet1D(in_channels=1, out_channels=3, base_channels=32).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 15
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
        )

    plot_example(model, val_dataset, device, index=0)


if __name__ == "__main__":
    main()