import torch
from src.config import Config
from src.model import SpanBERTCorefModel
import os

from transformers import AutoTokenizer
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
test_dataloader = CorefDataset(tokenizer, config, "test")

sent = "The president ate his dinner in the oval room."
inputs = tokenizer(sent, return_tensors="pt")

with torch.no_grad():
    inputs = {key: value.to(device) for key, value in inputs.items()}
    output, _ = model(**inputs)
    # Process the output as needed
    print(f"Output for example {sent}: {output}")