class MultilingualContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 0.5, temperature: float = 0.07, hard_negative_ratio: float = 0.5):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
        self.hard_negative_ratio = hard_negative_ratio
        
    def compute_similarity(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Computes the cosine similarity between two sets of given embeddings."""
        x1 = F.normalize(x1, p=2, dim=-1)
        x2 = F.normalize(x2, p=2, dim=-1)
        similarity = torch.matmul(x1, x2.transpose(-2, -1))
        return similarity
        
    def get_hard_negatives(self, sim_matrix: torch.Tensor, pos_mask: torch.Tensor) -> torch.Tensor:
        neg_mask = 1 - pos_mask
        neg_sim = sim_matrix * neg_mask
        neg_sim_sorted, _ = torch.sort(neg_sim, dim=1, descending=True)
        k = int(neg_sim_sorted.size(1) * self.hard_negative_ratio)
        hard_negatives = neg_sim_sorted[:, :k]
        
        return hard_negatives
        
    def forward(self, video_emb: torch.Tensor,text_embs: Dict[str, torch.Tensor],video_ids: List[str]) -> torch.Tensor:
        """
        Computes ethe contrastive loss between video and the given text embeddings.
        
        """
        batch_size = video_emb.size(0)
        device = video_emb.device
        total_contrastive_loss = torch.tensor(0.0, device=device)
        total_triplet_loss = torch.tensor(0.0, device=device)
        
        for lang, text_emb in text_embs.items():
            sim_matrix = self.compute_similarity(video_emb, text_emb) / self.temperature
            pos_mask = torch.eye(batch_size, device=device)          
            exp_sim = torch.exp(sim_matrix)
            log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
            contrastive_loss = (-log_prob * pos_mask).sum(dim=1).mean()           
            hard_negatives = self.get_hard_negatives(sim_matrix, pos_mask)
            pos_sim = torch.diagonal(sim_matrix)
            triplet_loss = torch.relu(
                self.margin - pos_sim.unsqueeze(1) + hard_negatives
            ).mean()      
            total_contrastive_loss += contrastive_loss
            total_triplet_loss += triplet_loss     
        num_langs = len(text_embs)
        final_loss = (total_contrastive_loss / num_langs) + (total_triplet_loss / num_langs)
        
        return final_loss 