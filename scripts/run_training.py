import time

import torch
from transformers import AutoTokenizer, AutoModel

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))


from src.config import Config
from src.data_loader import CorefDataset
from src.model import SpanBERTCorefModel
from src.train import train

if __name__ == '__main__':
    config = Config()

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    bert_model = AutoModel.from_pretrained(config.MODEL_NAME)

    dataloader = CorefDataset(tokenizer, config, "train")

    model = SpanBERTCorefModel(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    start_time = time.time()
    # Iterate through one batch from the DataLoader
    for idx,batch in enumerate(dataloader):
        batch = [item.to(device) for item in batch]

        optimizer.zero_grad()
        output = model(*batch)

        if idx % 10 == 0:
            print("Batch %d from %d" % (idx, len(dataloader)))

        if idx == 10:
            break


    print("Total Running time = {:.3f} seconds".format(time.time() - start_time))