import random
from pathlib import Path
import music21
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset


MIN_PITCH = 30
MAX_PITCH = 94
MIN_TRANSPOSE = -6
MAX_TRANSPOSE = 6
NUM_PARTS = 4
RESOLUTION = 4  # steps per quarter-note, e.g. 4 == 16th note resolution
WIDTH = 64
HEIGHT = MAX_PITCH - MIN_PITCH
BATCH_SIZE = 32


def train_val_test_dataloaders():
    ds = BachDataset()

    rng = random.Random(0)
    idx = list(range(len(ds)))
    rng.shuffle(idx)

    train_idx = idx[: int(len(ds) * 0.8)]
    val_idx = idx[len(train_idx) : int(len(ds) * 0.9)]
    test_idx = idx[len(train_idx) + len(val_idx) :]

    train_ds = Subset(ds, indices=train_idx)
    val_ds = Subset(ds, indices=test_idx)
    test_ds = Subset(ds, indices=val_idx)

    train_ds = TransformDataset(train_ds, transform=RandomCropAndTranspose())
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    return train_dl, val_dl, test_dl


class BachDataset(Dataset):
    def __init__(self):
        self.data = load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_data():
    cache_path = Path("dataset.cache.pt")
    if cache_path.exists():
        return torch.load(cache_path)

    data = []
    corpuses = music21.corpus.search("bach")
    for i, piece in enumerate(corpuses):
        if i % 10 == 0:
            print(f"{i+1}/{len(corpuses)}")
        piece = piece.parse()
        if len(piece.parts) == NUM_PARTS:
            data.append(piece_to_array(piece))
    torch.save(data, cache_path)
    return data


def piece_to_array(piece):
    duration = int(piece.expandRepeats().duration.quarterLength * RESOLUTION)
    arr = torch.zeros([NUM_PARTS, duration, MAX_PITCH - MIN_PITCH])

    for part_i, part in enumerate(piece.parts):
        notes = part.expandRepeats().flat.notes
        for note in notes:
            next_note = note.next("Note")
            next_is_same = next_note and note.pitch.midi == next_note.pitch.midi
            subtract = 1 if next_is_same else 0
            start_column = int(note.offset * RESOLUTION)
            end_column = int(
                (note.offset + note.duration.quarterLength) * RESOLUTION - subtract
            )
            pitch_row = note.pitch.midi - MIN_PITCH
            arr[part_i, start_column:end_column, pitch_row] = 1

    return arr


class RandomCropAndTranspose(nn.Module):
    def forward(self, arr):
        duration = arr.shape[1]
        t = torch.randint(0, duration - WIDTH, size=(1,)).item()
        t = 4 * (t // 16)
        cropped = arr[:, t : t + WIDTH]
        transpose = torch.randint(MIN_TRANSPOSE, MAX_TRANSPOSE, size=(1,)).item()
        transposed = torch.roll(cropped, transpose, dims=2)
        return transposed


class TransformDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x

    def __len__(self):
        return len(self.subset)
