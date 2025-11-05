
import torch
@torch.no_grad()
def recall_at_k(scores: torch.Tensor, labels: torch.Tensor, k: int):
    topk = torch.topk(scores, k=min(k, scores.shape[1]), dim=-1).indices
    hits = labels.gather(1, topk).sum(dim=1)
    return hits.float().mean().item()
