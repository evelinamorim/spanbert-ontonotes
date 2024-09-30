import time

import torch
from torch.optim.lr_scheduler import StepLR
from transformers import AutoTokenizer, AutoModel

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))


from src.config import Config
from src.data_loader import CorefDataset
from src.model import SpanBERTCorefModel
from src.train import train
from accelerate import Accelerator

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"


if __name__ == '__main__':
    config = Config()

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    bert_model = AutoModel.from_pretrained(config.MODEL_NAME)

    dataloader = CorefDataset(tokenizer, config, "train")

    model = SpanBERTCorefModel(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    accelerator = Accelerator()
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    model.train()

    accumulation_steps = 4

    start_time = time.time()
    # Iterate through one batch from the DataLoader
    for idx,batch in enumerate(dataloader):
        batch = tuple(item.to(device) if torch.is_tensor(item) else item for item in batch)

        #if idx % 10 == 0:
        print("Batch %d from %d" % (idx, len(dataloader)))

        with accelerator.autocast():  # Mixed precision context
            output, loss_batch = model(*batch)

            loss = loss_batch / accumulation_steps  # Scale loss

        accelerator.backward(loss)

        if (idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

        print(f"Loss: {loss.item()}")
        if idx == 10:
            torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_PATH,f"model_checkpoint_{idx}.pt"))
            break




    print("Total Running time = {:.3f} seconds".format(time.time() - start_time))
