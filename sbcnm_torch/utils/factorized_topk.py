
import torch
import torch.nn.functional as F

try:
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

class BruteForceRetriever:
    def __init__(self, item_embs: torch.Tensor, item_ids: torch.Tensor):
        self.item_embs = F.normalize(item_embs, p=2, dim=-1)
        self.item_ids = item_ids

    @torch.no_grad()
    def query(self, q: torch.Tensor, k=10):
        q = F.normalize(q, p=2, dim=-1)
        sims = q @ self.item_embs.T
        topk_vals, topk_idx = torch.topk(sims, k=min(k, self.item_embs.shape[0]), dim=-1)
        topk_ids = self.item_ids[topk_idx]
        return topk_vals, topk_ids

class FaissRetriever:
    def __init__(self, item_embs: torch.Tensor, item_ids: torch.Tensor):
        if not _HAS_FAISS:
            raise RuntimeError("faiss not available; install faiss-cpu")
        item_embs = F.normalize(item_embs, p=2, dim=-1).cpu().numpy().astype("float32")
        self.index = faiss.IndexFlatIP(item_embs.shape[1])
        self.index.add(item_embs)
        self.item_ids = item_ids.cpu().numpy().astype("int64")

    @torch.no_grad()
    def query(self, q: torch.Tensor, k=10):
        q = F.normalize(q, p=2, dim=-1).cpu().numpy().astype("float32")
        D, I = self.index.search(q, k)
        ids = torch.from_numpy(self.item_ids[I])
        sims = torch.from_numpy(D)
        return sims, ids
