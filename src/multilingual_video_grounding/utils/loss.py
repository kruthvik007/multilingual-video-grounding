import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

class MultilingualContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 0.5, temperature: float = 1.0):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
        
    def compute_similarity(self, x, y):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        return torch.matmul(x, y.t()) / self.temperature
    
    def get_hard_negatives(self, similarities, labels, k: int = 10):
        mask = ~labels.bool()
        hard_negatives = []
        
        for i in range(similarities.size(0)):
            neg_scores = similarities[i][mask[i]]
            if len(neg_scores) > 0:
                k_actual = min(k, len(neg_scores))
                hard_negatives.append(neg_scores.topk(k_actual, largest=True)[0])
                if k_actual < k:
                    padding = torch.zeros(k - k_actual, device=similarities.device)
                    hard_negatives[-1] = torch.cat([hard_negatives[-1], padding])
            else:
                hard_negatives.append(torch.randn(k, device=similarities.device) * 0.1)
        
        return torch.stack(hard_negatives)
    
    def forward(self, outputs, temperatures):
        video_emb = outputs['video_embeddings']
        text_embs = outputs['text_embeddings']
        total_loss = 0.0
        num_pairs = 0
        
        for lang1 in text_embs.keys():
            for lang2 in text_embs.keys():
                if lang1 != lang2:
                    num_pairs += 1
                    temp = temperatures[f"{lang1}_{lang2}"]
                    video_text_sim = self.compute_similarity(video_emb, text_embs[lang1])
                    text_text_sim = self.compute_similarity(text_embs[lang1], text_embs[lang2])
                    labels = torch.eye(video_emb.size(0), device=video_emb.device)
                    hard_neg_video = self.get_hard_negatives(video_text_sim, labels)
                    hard_neg_text = self.get_hard_negatives(text_text_sim, labels)
                    pos_sim_video = torch.diag(video_text_sim).unsqueeze(1)
                    all_sim_video = torch.cat([pos_sim_video, hard_neg_video], dim=1)
                    video_text_loss = -pos_sim_video + torch.logsumexp(all_sim_video, dim=1) + self.margin
                    pos_sim_text = torch.diag(text_text_sim).unsqueeze(1)
                    all_sim_text = torch.cat([pos_sim_text, hard_neg_text], dim=1)
                    text_text_loss = -pos_sim_text + torch.logsumexp(all_sim_text, dim=1) + self.margin
                    pair_loss = (video_text_loss.mean() + text_text_loss.mean()) * temp
                    total_loss += pair_loss
        
        if num_pairs > 0:
            total_loss = total_loss / num_pairs
        
        return total_loss
