print("hello world")
from nanochat.dataloader import tokenizing_distributed_data_loader

B, T = 4, 16
dl = tokenizing_distributed_data_loader(B, T, split="train", tokenizer_threads=1, tokenizer_batch_size=64, device="cpu")

src, tgt = next(dl)
print(src)
print(tgt)
