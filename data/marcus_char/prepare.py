# ~/Transformers/nanoGPT/data/marcus_char/prepare.py

import os
import pickle
import numpy as np

# 1) Load your raw text
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()

print(f"Length of dataset in characters: {len(data):,}")

# 2) Extract unique characters, build vocab
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("All unique characters:", ''.join(chars))
print(f"Vocab size: {vocab_size}")

# 3) Create char-to-int and int-to-char mappings
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

# 4) Train/val split (90/10, for example)
n = len(data)
train_data = data[:int(n*0.9)]
val_data   = data[int(n*0.9):]

# 5) Encode both splits
train_ids = encode(train_data)
val_ids   = encode(val_data)
print(f"Train has {len(train_ids):,} tokens")
print(f"Val has {len(val_ids):,} tokens")

# 6) Save to .bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids   = np.array(val_ids,   dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# 7) Save meta info
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print("Data preparation complete!")
