import torch
import os
import sys

from transformers import AutoTokenizer

from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))

from src.config import Config
from src.model import SpanBERTCorefModel
from src.data_loader import CorefDataset

# Initialize the config and model
config = Config()
model = SpanBERTCorefModel(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
model.to(device)

# Load the saved checkpoint
checkpoint_path = os.path.join(config.CHECKPOINT_PATH, "model_checkpoint_10.pt")
model.load_state_dict(torch.load(checkpoint_path))
model.eval()


tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
dataset = CorefDataset(tokenizer, config, "adhoc")

sent = ["[CLS]","the", "president","ate","his","dinner","in","the","oval", "room",".","[SEP]"]
subtoken_map = []
word_idx = -1
for word in sent:
    subtokens = tokenizer.tokenize(word)
    word_idx += 1
    for sidx, subtokens in enumerate(subtokens):
        subtoken_map.append(word_idx)

item = {}
item["doc_key"] = "bc/cctv/00/cctv_000_0"
item["sentences"] = [sent]
item["clusters"] = [[[1,3]]]
item["speakers"] = [["#Speaker1","#Speaker1","#Speaker1","#Speaker1","#Speaker1","#Speaker1","#Speaker1","#Speaker1","#Speaker1","#Speaker1","#Speaker1","#Speaker1"]]
item["subtoken_map"] = subtoken_map
item["sentence_map"] = [0,0,0,0,0,0,0,0,0,0,0,0]

with torch.no_grad():
    inputs = dataset.sentence_to_tensor(item)

    output, _ = model(*inputs)
    # Process the output as needed
    print(f"Output for example {sent}: {output}")
