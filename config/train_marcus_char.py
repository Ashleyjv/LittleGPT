# ~/Transformers/nanoGPT/config/train_marcus_char.py

out_dir = 'out-marcus-char'
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = 'marcus-char'
wandb_run_name = 'mini-gpt'

dataset = 'marcus_char'  # <-- points to your new data folder
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100

# if you want to disable compile:
compile = False
