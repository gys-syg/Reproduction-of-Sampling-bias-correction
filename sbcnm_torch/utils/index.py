
import torch
import torch.nn.functional as F
def build_item_index(item_tower, item_feats, batch_size=1024, device="cpu"):
    item_tower.eval()
    embs = []
    with torch.no_grad():
        for i in range(0, item_feats.shape[0], batch_size):
            x = item_feats[i:i+batch_size].to(device)
            e = item_tower(x)
            embs.append(e.cpu())
    embs = torch.cat(embs, dim=0)
    embs = F.normalize(embs, p=2, dim=-1)
    return embs
