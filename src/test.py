import torch
import os
import sys

from transformers import AutoTokenizer

from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
from src.config import Config
from src.model import SpanBERTCorefModel

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

sent = "The president ate his dinner in the oval room."
inputs = tokenizer(sent, return_tensors="pt")

with torch.no_grad():
    new_inputs = {}
    new_inputs["input_ids"] = inputs["input_ids"].to(device)
    new_inputs["input_mask"] = inputs["attention_mask"].to(device)
    new_inputs["text_len"] = 1
    new_inputs["speaker_ids"] = torch.tensor([0,1])
    new_inputs["genre"] = "bc"
    new_inputs["is_training"] = False
    new_inputs["gold_starts"] = torch.tensor([1,3])
    new_inputs["gold_ends"] = torch.tensor([1,3])
    new_inputs["cluster_ids"] = torch.tensor([1])
    new_inputs["sentence_map"] = torch.tensor([0,0,0,0,0,0,0,0,0,0,0,0])

    output, _ = model(**new_inputs)
    # Process the output as needed
    print(f"Output for example {sent}: {output}")
