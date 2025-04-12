import os
import pickle
import torch
import tiktoken
from contextlib import nullcontext
from model import GPTConfig, GPT

# ----------------------------------------------------------------------
# Basic settings
# ----------------------------------------------------------------------
CKPT_PATH = "out-shakespeare-char/ckpt.pt"  # or wherever your checkpoint is
DEVICE = "cuda"
DTYPE = "float16"  # or "bfloat16" if your GPU supports it
MAX_NEW_TOKENS = 100
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
#   - For Shakespeare char model, we have a "meta.pkl"
#   - If you trained a GPT-2 BPE model, you'd load GPT-2 encodings
# ----------------------------------------------------------------------
meta_path = None
if 'config' in checkpoint and 'dataset' in checkpoint['config']:
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')

if meta_path and os.path.exists(meta_path):
    # If we have a character-level Shakespeare meta
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi.get(c, 0) for c in s]  # unknown chars to 0
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # Otherwise assume GPT-2 encodings
    print("No meta.pkl found (or not a char-level model). Assuming GPT-2 encodings.")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# ----------------------------------------------------------------------
# Simple chat loop
# ----------------------------------------------------------------------

print("=== nanoGPT Chat Demo ===")
print("Type something and press Enter. Type 'exit' or Ctrl+C to quit.\n")

# You can store a running 'conversation' in a string if you want to keep context
conversation = ""

while True:
    # 1) Get user input
    user_input = input("User: ")
    if user_input.strip().lower() == 'exit':
        print("Exiting chat.")
        break

    # 2) Append user input to conversation
    conversation += user_input

    # 3) Encode the conversation, generate
    with torch.no_grad():
        with ctx:
            # Encode current conversation
            x = torch.tensor([encode(conversation)], dtype=torch.long, device=DEVICE)
            # Generate
            y = model.generate(
                x,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_k=TOP_K
            )
    # The model’s entire output (including the prompt)
    generated_tokens = y[0].tolist()
    full_text = decode(generated_tokens)

    # 4) Extract the newly generated part (everything after the original conversation length)
    new_text = full_text[len(conversation):]

    # 5) Print the model's reply
    print(f"Model: {new_text.strip()}")
    print("---------------")

    # 6) Append the model reply to the conversation if you want continuing context
    conversation += new_text
