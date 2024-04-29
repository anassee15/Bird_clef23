# basic packages
import os
import json
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt

# Deep Learning framework
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Audio processing
import torchaudio
import torchaudio.transforms as T

# Image processing
from PIL import Image
import torchvision.transforms as transforms

# Pre-trained image models
import timm


class BirdDataset(Dataset):
    def __init__(self, df, sample_rate=32000):
        self.df = df
        self.file_paths = df["file_path"].values
        self.target = df["label"].values
        self.labels = df["encoded_label"].values
        self.sample_rate = sample_rate
        self.melspectrogram = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=128,
            n_fft=2048,
            hop_length=512,
            f_min=500,
            f_max=15000,
        )
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio, sr = torchaudio.load(self.file_paths[idx])

        # if audio has more than one channel, convert it to mono by averaging the channels
        audio = torch.mean(audio, axis=0)

        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            audio = resampler(audio)

        # add random noise to audio
        if np.random.rand() < 0.3:
            noise = torch.randn_like(audio) * 0.005
            audio += noise

        # normalize audio
        audio = audio / torch.max(torch.abs(audio))

        # convert audio to melspectrogram
        melspec = self.melspectrogram(audio)

        # convert melspec to image
        image = Image.fromarray(melspec.numpy()).convert("RGB")

        return {
            "image": self.preprocess(image),
            "label": torch.Tensor(self.labels[idx]),
            "label_text": self.target[idx],
        }


class BirdModel(nn.Module):
    def __init__(
        self, model_name="tf_efficientnet_b3", pretrained=True, num_classes=264
    ):
        super(BirdModel, self).__init__()

        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(nn.Linear(self.in_features, num_classes))

    def forward(self, img):
        logits = self.model(img)
        return logits


DATA_PATH = os.path.join("..", "data", "birdclef-2023")
TRAIN_PATH = os.path.join(DATA_PATH, "train_audio_split")

X = []
y = []

for dirname, _, filename in os.walk(TRAIN_PATH):
    for file in filename:
        X.append(os.path.join(dirname, file))
        y.append(
            os.path.join(dirname, file)
            .replace(TRAIN_PATH + os.sep, "")
            .split(os.sep)[0]
        )

# create pandas dataframe
df = pd.DataFrame({"file_path": X, "label": y})


def remove_overrepresented_label(df, label, size=0.5):
    df_label = df.loc[df["label"] == label]
    half_size = int(len(df_label) * size)
    df_label_sampled = df_label.sample(n=half_size)
    return df.drop(df_label_sampled.index)


def augment_underrepresented_label(df, label, size=2):
    df_label = df[df["label"] == label]
    num_samples = int(size * len(df_label) - len(df_label))
    df_oversampled = df_label.sample(n=num_samples, replace=True)
    df_augmented = pd.concat([df, df_oversampled], ignore_index=True)
    return df_augmented


groupby_label = (
    df.groupby("label")
    .count()
    .reset_index()
    .sort_values(by="file_path", ascending=False)
)


label_to_remove = groupby_label[groupby_label["file_path"] > 10000]["label"].values

for label in label_to_remove:
    df = remove_overrepresented_label(df, label)

label_to_augment = groupby_label[groupby_label["file_path"] < 25]["label"].values

for label in label_to_augment:
    df = augment_underrepresented_label(df, label, size=4)

label_to_augment = groupby_label[groupby_label["file_path"] < 50]["label"].values

for label in label_to_augment:
    df = augment_underrepresented_label(df, label, size=2)

label_to_augment = groupby_label[groupby_label["file_path"] < 75]["label"].values

for label in label_to_augment:
    df = augment_underrepresented_label(df, label, size=1.5)


# one hot encoding
# get unique label sorted by name
unique_labels = sorted(df["label"].unique())

# create dictionary with label as key and encoded label as value
label_to_id = {label: i for i, label in enumerate(unique_labels)}

for k in label_to_id.keys():
    label_to_id[k] = torch.nn.functional.one_hot(
        torch.tensor(label_to_id[k]), num_classes=len(unique_labels)
    ).tolist()

# save one hot encoded labels in a json file
with open("one_hot_encoding.json", "w") as f:
    json.dump(label_to_id, f)

# load one hot encoded labels from json file and add it to dataframe
with open("one_hot_encoding.json", "r") as f:
    label_to_id = json.load(f)

df["encoded_label"] = df["label"].map(label_to_id)

# split train and validation set
train_df, val_df = train_test_split(
    df, test_size=0.15, random_state=42, stratify=df["label"], shuffle=True
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

train_df.shape, val_df.shape

# create dataset
train_dataset = BirdDataset(train_df)
val_dataset = BirdDataset(val_df)

# create dataloader
train_loader = DataLoader(train_dataset, batch_size=16)
val_loader = DataLoader(val_dataset, batch_size=16)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# create model
model = BirdModel()
model.to(device)

print(device)
print(model)

EPOCHS = 10
BEST_MODEL_PATH = os.path.join("..", "model", "best_bird_model.pth")

# parameters
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
criterion = nn.CrossEntropyLoss()

# history
train_loss_history, val_loss_history = [], []
train_accuracy_history, val_accuracy_history = [], []
best_accuracy = 0.0


def save_plot(
    train_loss_history, val_loss_history, train_accuracy_history, val_accuracy_history
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    ax1.plot(train_loss_history, label="train")
    ax1.plot(val_loss_history, label="val")
    ax1.set_title("Loss history")
    ax1.legend()

    ax2.plot(train_accuracy_history, label="train")
    ax2.plot(val_accuracy_history, label="val")
    ax2.set_title("Accuracy history")
    ax2.legend()

    plt.savefig(os.path.join("assets", "history.png"))


# train
model.train()

for _ in range(EPOCHS + 1):
    print(f"EPOCH: {_+1}/{EPOCHS}")
    train_loss = 0.0
    train_error_count = 0.0

    for data in train_loader:
        optimizer.zero_grad()
        X = data["image"].to(device)
        y = data["label"].to(device)

        outputs = model(X)

        loss = criterion(outputs, y)
        train_loss += loss.item()
        train_error_count += (outputs.argmax(1) != y.argmax(1)).sum().item()
        loss.backward()
        optimizer.step()

    train_loss /= len(train_loader)
    val_loss = 0.0
    val_error_count = 0.0

    for data in val_loader:
        X = data["image"].to(device)
        y = data["label"].to(device)

        outputs = model(X)

        loss = criterion(outputs, y)
        val_loss += loss.item()
        val_error_count += (outputs.argmax(1) != y.argmax(1)).sum().item()

    val_loss /= len(val_loader)

    train_accuracy = 1.0 - float(train_error_count) / len(train_dataset)
    val_accuracy = 1.0 - float(val_error_count) / len(val_dataset)

    train_loss_history.append(train_loss)
    train_accuracy_history.append(train_accuracy)

    val_loss_history.append(val_loss)
    val_accuracy_history.append(val_accuracy)

    # save best model
    if val_accuracy > best_accuracy:
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        best_accuracy = val_accuracy

    # save plot
    save_plot(
        train_loss_history,
        val_loss_history,
        train_accuracy_history,
        val_accuracy_history,
    )

model.eval()

model = BirdModel()
model.load_state_dict(torch.load(BEST_MODEL_PATH))
model.to(device)
model.eval()

y_pred = []
y_true = []

with torch.no_grad():
    for data in val_loader:

        X = data["image"].to(device)
        y = data["label"]

        outputs = model(X)

        y_true.extend(torch.argmax(outputs, dim=1).detach().cpu().numpy())
        y_pred.extend(torch.argmax(y, dim=1).numpy())

with open(os.path.join("assets","metrics.txt"), "w") as f:
    f.write(f"Accuracy: {accuracy_score(y_true, y_pred)}\n")
    print(f"Accuracy: {accuracy_score(y_true, y_pred)}\n")
    f.write(f"F1: {f1_score(y_true, y_pred, average='macro')}\n")
    print(f"F1: {f1_score(y_true, y_pred, average='macro')}\n")
    f.write(f"Precision: {precision_score(y_true, y_pred, average='macro')}\n")
    print(f"Precision: {precision_score(y_true, y_pred, average='macro')}\n")
    f.write(f"Recall: {recall_score(y_true, y_pred, average='macro')}\n")
    print(f"Recall: {recall_score(y_true, y_pred, average='macro')}\n")
