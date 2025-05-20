import argparse
import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random

# 设定随机种子
fix_seed = 2024
random.seed(fix_seed)
np.random.seed(fix_seed)

# 参数解析
parser = argparse.ArgumentParser(description="WFlib Unified")
parser.add_argument("--dataset", type=str, default="Palette", help="Dataset name")
parser.add_argument("--use_stratify", type=str, default="True", help="Whether to use stratify when splitting")
args = parser.parse_args()

# 输入输出路径
in_path = '/mnt/e/Dataset/KDataset/DF-Palette' # 从wsl调用windows文件/mnt/e/
dataset_path = os.path.join("../npz_dataset", args.dataset)
os.makedirs(dataset_path, exist_ok=True)

filenames = glob.glob(in_path + "/*")

# ---------- 工具函数 ----------
def pad_sequence(sequence, length):
    if len(sequence) >= length:
        sequence = sequence[:length]
    else:
        sequence = np.pad(sequence, (0, length - len(sequence)), "constant", constant_values=0.0)
    return sequence

def parse_file(files):
    file_labels = []
    file_sequence = []
    for file in tqdm(files):
        if not os.path.getsize(file) > 0:
            continue

        filename = os.path.basename(file)
        if "-" not in filename:
            file_labels.append(-1)
        else:
            file_labels.append(int(filename.split("-")[0]))

        with open(file, 'r') as f:
            tmp = f.readlines()
        data = pd.Series(tmp).str.slice(0, -1).str.split('\t', expand=True).astype('float').to_numpy()

        timestamp = pad_sequence(data[:, 0], 10000)
        timestamp = np.trim_zeros(timestamp, "b")
        timestamp[timestamp == 0.0] = 1e-6
        timestamp = np.pad(timestamp, (0, 10000 - len(timestamp)), mode='constant', constant_values=0)

        timestamp = timestamp.reshape((1, -1, 1))
        packet_length = pad_sequence(data[:, 1], 10000).reshape((1, -1, 1))
        packet_data = np.concatenate([timestamp, packet_length], axis=-1)
        packet_data = packet_data.astype(np.float32)

        file_sequence.append(packet_data)

    return np.concatenate(file_sequence, axis=0), np.array(file_labels)

# ---------- 数据解析 ----------
print("Parsing raw files...")
X, labels = parse_file(filenames)
max_category = np.max(labels[labels != -1])
labels[labels == -1] = max_category + 1
labels = labels.astype(np.uint8)

# 保存完整数据
print(f"All: X = {X.shape}, y = {labels.shape}")
np.savez_compressed(os.path.join(dataset_path, "data.npz"), X=X, y=labels)

# ---------- 数据集划分 ----------
print("Splitting dataset...")
num_classes = len(np.unique(labels))
assert num_classes == labels.max() + 1, "Labels are not continuous"
print(f"Num classes: {num_classes}")
print(f"Unique labels: {list(np.unique(labels))}")

if args.use_stratify == "True":
    X_train, X_test, y_train, y_test = train_test_split(X, labels, train_size=0.9, random_state=fix_seed, stratify=labels)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.9, random_state=fix_seed, stratify=y_train)
else:
    X_train, X_test, y_train, y_test = train_test_split(X, labels, train_size=0.9, random_state=fix_seed)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.9, random_state=fix_seed)

# ---------- 保存划分后的数据 ----------
print(f"Train: X = {X_train.shape}, y = {y_train.shape}")
print(f"Valid: X = {X_valid.shape}, y = {y_valid.shape}")
print(f"Test:  X = {X_test.shape}, y = {y_test.shape}")

np.savez_compressed(os.path.join(dataset_path, "train.npz"), X=X_train, y=y_train)
np.savez_compressed(os.path.join(dataset_path, "valid.npz"), X=X_valid, y=y_valid)
np.savez_compressed(os.path.join(dataset_path, "test.npz"), X=X_test, y=y_test)
