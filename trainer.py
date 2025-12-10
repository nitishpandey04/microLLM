from dataclasses import dataclass
from modeling import MicroLLM, MicroLLMConfig
from typing import Generator
import torch.nn.functional as F
from torch.optim import AdamW
from nanochat.dataloader import tokenizing_distributed_data_loader

@dataclass
class TrainArgs:
    num_steps: 5000
    batch_size: int = 32
    max_seq_len: int = 512
    lr: float = 1e-5
    label_smoothing: float = 0.01
    device: str = "cuda"

class Trainer:
    def __init__(self, model: MicroLLM, args: TrainArgs) -> None:
        self.model = model
        self.args = args
        self.optimizer = AdamW(model.parameters(), lr=args.lr)
        self._prepare_dataloader()

    def train(self):
        for step, (src, tgt) in enumerate(self.dataloader):
            if step == self.args.num_steps:
                break
                
            B, T = src.shape
            logits = self.model(src)

            loss = F.cross_entropy(
                logits.view(B * T, -1),
                tgt.view(B * T,),
                label_smoothing=self.args.label_smoothing
            )
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if step % 100 == 0:
                print(f"step: {step + 1:2d} | loss: {loss:.5f}")
        
    def _save_checkpoint(self):
        pass
        
    def _load_checkpoint(self):
        pass
        
    def _prepare_dataloader(self):
        self.dataloader = tokenizing_distributed_data_loader(
            B=self.args.batch_size,
            T=self.args.max_seq_len,
            split="train",
            tokenizer_threads=4,
            tokenizer_batch_size=64,
            device=self.args.device
        )
        