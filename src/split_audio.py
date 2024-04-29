import pandas as pd
import soundfile as sf
import librosa
import os


DATA_PATH = os.path.join("..", "data", "birdclef-2023")
TRAIN_PATH = os.path.join(DATA_PATH, "train_audio")
SAVE_PATH = os.path.join(DATA_PATH, "train_audio_split")


def split_audio(file_path, save_path, duration=5):
    # load audio file with librosa
    audio, sr = librosa.load(file_path, sr=None)

    # compute duration
    duration_samples = int(duration * sr)
    segments = range(0, len(audio), duration_samples)

    i = 5
    # save each segment as a wav file
    for start in segments:
        # save only if the audio is equal than the duration
        if len(audio[start:]) >= duration_samples:
            path = save_path.split(os.sep)
            path[-1] = path[-1].replace(".", f"_{i}.")
            path = os.sep.join(path)
            segment = audio[start : start + duration_samples]
            sf.write(path, segment, sr)
            i += 5
        else:
            break


if __name__ == "__main__":
    file_id = []

    for dirname, _, filename in os.walk(TRAIN_PATH):
        for file in filename:
            file_id.append(os.path.join(dirname, file))

    train = pd.DataFrame()
    train = train.assign(filename=file_id)
    train["label"] = train["filename"].apply(
        lambda x: x.replace(TRAIN_PATH + os.sep, "").split(os.sep)[0]
    )

    for file, label in zip(train["filename"], train["label"]):
        os.makedirs(os.path.join(SAVE_PATH, label), exist_ok=True)
        split_audio(file, os.path.join(SAVE_PATH, label, (file.split(os.sep)[-1])))
