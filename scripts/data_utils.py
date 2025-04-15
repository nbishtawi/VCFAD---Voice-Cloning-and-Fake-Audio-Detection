import os
import shutil
from pathlib import Path

def rename_and_move_audio_files(src_folder, dest_folder, ext=".wav"):
    """Rename and move all audio files into a single directory with consistent names."""
    os.makedirs(dest_folder, exist_ok=True)
    count = 0
    for root, _, files in os.walk(src_folder):
        for file in files:
            if file.endswith(ext):
                count += 1
                new_name = f"audio_{count:04d}{ext}"
                shutil.copy(os.path.join(root, file), os.path.join(dest_folder, new_name))
    print(f"Moved and renamed {count} files to {dest_folder}")

def split_dataset(audio_folder, output_dir, train_ratio=0.8):
    """Split audio files into train/test sets."""
    os.makedirs(output_dir, exist_ok=True)
    all_files = sorted([f for f in os.listdir(audio_folder) if f.endswith(".wav")])
    split_idx = int(len(all_files) * train_ratio)
    train_files = all_files[:split_idx]
    test_files = all_files[split_idx:]

    for split, files in zip(["train", "test"], [train_files, test_files]):
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        for f in files:
            shutil.copy(os.path.join(audio_folder, f), os.path.join(split_dir, f))

    print(f"Split complete. {len(train_files)} train files, {len(test_files)} test files.")
