# Utility functions, such as for logging, checkpoint saving, and other helper methods.
import torch


def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)

# original from:https://github.com/WenzhengZhang/Seq2seqCoref/blob/main/utils.py

def flat_lists(ls):
    return [l for s in ls for l in s]