# Utility functions, such as for logging, checkpoint saving, and other helper methods.
import torch


def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)

# original from:https://github.com/WenzhengZhang/Seq2seqCoref/blob/main/utils.py

def flat_lists(ls):
    return [l for s in ls for l in s]

def batch_gather(emb, indices):
    batch_size, seqlen = emb.size(0), emb.size(1)
    emb_size = emb.size(2) if emb.dim() > 2 else 1

    flattened_emb = emb.view(batch_size * seqlen, emb_size)  # [batch_size * seqlen, emb]
    offset = (torch.arange(batch_size) * seqlen).unsqueeze(1).to(indices.device)  # [batch_size, 1]

    adjusted_indices = indices + offset
    gathered_embeddings = flattened_emb[adjusted_indices.view(-1)]  
    gathered_embeddings = gathered_embeddings.view(batch_size, indices.size(1), -1) # [batch_size, num_indices, emb]


    if emb.dim() == 2:
        gathered_embeddings = gathered_embeddings.squeeze(2)  # [batch_size, num_indices]

    return gathered_embeddings
