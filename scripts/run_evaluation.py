import torch

from src.config import Config
from src.data_loader import load_data
from src.model import SpanBERTCorefModel
from src.evaluation import evaluate

config = Config()
_, val_dataloader = load_data(config)
model = SpanBERTCorefModel(config)
model.load_state_dict(torch.load("saved_models/fine_tuned_model/model.pth"))
evaluate(model, val_dataloader, config)
