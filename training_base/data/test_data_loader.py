# test_dataset.py
from datasets import SlakhDataset
PATH = "/Users/jethro.r.lee/Documents/NYU/Music Information Retrieval/mir-final-project-instrument-classification/training_base/data/babyslakh_16k"
import soundfile as sf
from IPython.display import Audio

dataset = SlakhDataset(
    split_cfg={"root": PATH, "split": "none"},
    task_type="instrument_classification",
    num_classes=16,
    sample_rate=16000,
    clip_num_samples=16000 * 4,
    train_mode=False,
    manipulate=True,
)

print(f"Dataset size: {len(dataset)}")

audio, label = dataset[1]
print(f"Audio shape: {audio.shape}")  # expect torch.Size([88200])
print(f"Label: {label.item()}")       # expect 0-15

# Play the signal
sound = audio.numpy()   # convert torch tensor → numpy
sr = 16000              # matches your dataset config
sf.write("sample.wav", sound, 16000)

print(Audio(sound, rate=sr))
