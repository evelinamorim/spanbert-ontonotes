import time

import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).absolute().parent.parent))

from src.config import Config
from src.data_loader import CorefDataset
from src.model import SpanBERTCorefModel
from accelerate import Accelerator, DistributedDataParallelKwargs
from src.data_loader import collate_fn

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

if __name__ == '__main__':
    config = Config()

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    bert_model = AutoModel.from_pretrained(config.MODEL_NAME)

    mydataset = CorefDataset(tokenizer, config, "train")

    train_dataloader = DataLoader(mydataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn)
    
    model = SpanBERTCorefModel(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(gradient_accumulation_steps=4, mixed_precision="fp16",
                              kwargs_handlers=[ddp_kwargs])
    device = accelerator.device
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, train_dataloader, scheduler)
    
    model.train()
    
    start_time = time.time()
    # # Iterate through one batch from the DataLoader
    for idx, batch in enumerate(train_dataloader):
    #     #if idx % 10 == 0:
        print("Batch %d from %d" % (idx, len(train_dataloader)))
        batch_size = batch[0].size(0)  # ou o tamanho relevante do batch
        print(f"Batch size in device {accelerator.device}: {batch_size}\n")
    
        with accelerator.autocast():  # Mixed precision context
             output, loss_batch = model(*batch)
        break
    #
    #         loss = loss_batch / accelerator.gradient_accumulation_steps  # Scale loss
    #
    #     accelerator.backward(loss)
    #
    #     if (idx + 1) % accelerator.gradient_accumulation_steps == 0:
    #         optimizer.step()
    #         optimizer.zero_grad()
    #         torch.cuda.empty_cache()
    #
    #     print(f"Loss: {loss.item()}")
    #     if idx == 10:
    #         #torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_PATH,f"model_checkpoint_{idx}.pt"))
    #         accelerator.wait_for_everyone()
    #         accelerator.save_model(model, os.path.join(config.CHECKPOINT_PATH, f"model_checkpoint_{idx}.pt"))
    #         break
    #
    # print("Total Running time = {:.3f} seconds".format(time.time() - start_time))
