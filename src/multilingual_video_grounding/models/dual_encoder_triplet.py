import torch
import torch.nn as nn
from transformers import XLMRobertaModel, BertModel
from typing import Dict, List, Tuple
import torch.nn.functional as F

class ContentProjection(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.content_transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)
        
    def forward(self, x):
        return self.content_transform(x)

class VideoEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        self.content_projection = ContentProjection(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )
        
    def forward(self, x):
        x_attended, _ = self.temporal_attention(x, x, x)
        x_pooled = x_attended.mean(dim=1)
        return self.content_projection(x_pooled)

class LanguageEncoder(nn.Module):
    def __init__(self, model_name: str, output_dim: int, hidden_dim: int):
        super().__init__()
        if 'chinese' in model_name.lower():
            self.encoder = BertModel.from_pretrained('bert-base-chinese')
        else:
            self.encoder = XLMRobertaModel.from_pretrained(model_name)
        self.content_projection = ContentProjection(
            input_dim=self.encoder.config.hidden_size,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return self.content_projection(outputs.last_hidden_state[:, 0, :])

class TripletLoss(nn.Module):
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin
        
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        basic_loss = F.relu(pos_dist - neg_dist + self.margin)
        modality_reg = torch.mean(torch.abs(torch.mean(anchor, dim=0)))
        return basic_loss.mean() + 0.1 * modality_reg

class MultilingualTripletEncoder(nn.Module):
    def __init__(self,
                 video_input_dim: int = 1024,
                 hidden_dim: int = 512,
                 output_dim: int = 256,
                 languages: List[str] = ['chinese', 'german']):
        super().__init__()
        self.video_input_dim = video_input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.video_encoder = VideoEncoder(
            input_dim=video_input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )
        self.text_encoders = nn.ModuleDict({
            'chinese': LanguageEncoder('bert-base-chinese', output_dim, hidden_dim),
            'german': LanguageEncoder('xlm-roberta-base', output_dim, hidden_dim)
        })
        self.triplet_loss = TripletLoss(margin=0.2)
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
    def encode_video(self, video_features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.video_encoder(video_features), p=2, dim=-1)
    
    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, language: str) -> torch.Tensor:
        return F.normalize(self.text_encoders[language](input_ids, attention_mask), p=2, dim=-1)
    
    def compute_triplet_loss(self, 
                           anchor_emb: torch.Tensor,
                           pos_emb: torch.Tensor, 
                           neg_emb: torch.Tensor) -> torch.Tensor:
        return self.triplet_loss(anchor_emb, pos_emb, neg_emb)
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size = batch['video_features'].size(0)
        assert batch['chinese_ids'].size(0) == batch_size
        assert batch['german_ids'].size(0) == batch_size
        assert batch['neg_video_features'].size(0) == batch_size
        assert batch['neg_chinese_ids'].size(0) == batch_size
        assert batch['neg_german_ids'].size(0) == batch_size
        video_emb = self.encode_video(batch['video_features'])
        chinese_emb = self.encode_text(batch['chinese_ids'], batch['chinese_mask'], 'chinese')
        german_emb = self.encode_text(batch['german_ids'], batch['german_mask'], 'german')
        neg_video_emb = self.encode_video(batch['neg_video_features'])
        neg_chinese_emb = self.encode_text(batch['neg_chinese_ids'], batch['neg_chinese_mask'], 'chinese')
        neg_german_emb = self.encode_text(batch['neg_german_ids'], batch['neg_german_mask'], 'german')
        assert video_emb.size(0) == batch_size and video_emb.size(1) == self.output_dim
        assert chinese_emb.size(0) == batch_size and chinese_emb.size(1) == self.output_dim
        assert german_emb.size(0) == batch_size and german_emb.size(1) == self.output_dim
        loss_v_c = self.compute_triplet_loss(video_emb, chinese_emb, neg_chinese_emb)
        loss_v_g = self.compute_triplet_loss(video_emb, german_emb, neg_german_emb)
        loss_c_g = self.compute_triplet_loss(chinese_emb, german_emb, neg_german_emb)
        loss_c_v = self.compute_triplet_loss(chinese_emb, video_emb, neg_video_emb)
        loss_g_v = self.compute_triplet_loss(german_emb, video_emb, neg_video_emb)
        loss_g_c = self.compute_triplet_loss(german_emb, chinese_emb, neg_chinese_emb)
        total_loss = (loss_v_c + loss_v_g + loss_c_g + loss_c_v + loss_g_v + loss_g_c) / 6.0
        return {
            'loss': total_loss,
            'video_embeddings': video_emb,
            'chinese_embeddings': chinese_emb,
            'german_embeddings': german_emb
        }
