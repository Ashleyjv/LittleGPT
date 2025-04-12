import os
import pickle
import torch
import tiktoken
from contextlib import nullcontext
from model import GPTConfig, GPT

# ----------------------------------------------------------------------
# Basic settings - update as needed
# ----------------------------------------------------------------------
CKPT_PATH = "out-marcus-char/ckpt.pt"   # path to your trained checkpoint
DEVICE = "cuda"
DTYPE = "float16"     # could be "bfloat16" if supported
MAX_NEW_TOKENS = 500
TEMPERATURE = 0.7
TOP_K = 200

# ----------------------------------------------------------------------
# Load checkpoint
# ----------------------------------------------------------------------
checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(DEVICE)

ptdtype = {
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
    'float16': torch.float16
}[DTYPE]
device_type = 'cuda' if 'cuda' in DEVICE else 'cpu'
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# ----------------------------------------------------------------------
# Encoding/decoding
#   - This is a character-level model if meta.pkl is present
# ----------------------------------------------------------------------
meta_path = None
if 'config' in checkpoint and 'dataset' in checkpoint['config']:
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')

if meta_path and os.path.exists(meta_path):
    # Likely a char-level dataset
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    def encode(s):
        return [stoi.get(c, 0) for c in s]  # unknown chars -> 0
    def decode(l):
        return ''.join([itos[i] for i in l])
else:
    # Otherwise assume GPT-2 BPE
    print("No meta.pkl found. Using GPT-2 BPE encoding as fallback.")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# ----------------------------------------------------------------------
# Simple chat loop
# ----------------------------------------------------------------------
print("=== Chat with Marcus Aurelius (nanoGPT) ===")
print("Type something and press Enter. Type 'exit' or Ctrl+C to quit.\n")

conversation = ""

while True:
    user_input = input("User: ")
    if user_input.strip().lower() == 'exit':
        print("Exiting chat.")
        break

    # Append user input to conversation
    conversation += user_input

    # Generate a reply
    with torch.no_grad():
        with ctx:
            x = torch.tensor([encode(conversation)], dtype=torch.long, device=DEVICE)
            y = model.generate(
                x,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_k=TOP_K
            )
    # Decode the entire text
    generated_tokens = y[0].tolist()
    full_text = decode(generated_tokens)

    # The newly generated portion:
    new_text = full_text[len(conversation):]

    print(f"Marcus Aurelius: {new_text.strip()}")
    print("---------------")

    # Optionally continue the context for multi-turn conversation
    conversation += new_text
