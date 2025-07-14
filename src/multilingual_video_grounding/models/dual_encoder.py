import torch
import torch.nn as nn
from transformers import XLMRobertaModel, BertModel
from typing import Dict, List, Tuple
import torch.nn.functional as F
from ..utils.loss import MultilingualContrastiveLoss
from .projection_heads import DualProjection

class FeatureFusion(nn.Module):
    def __init__(self, input_dims: List[int], output_dim: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8,
            dropout=0.1
        )
        self.projection = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in input_dims
        ])
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, features: List[torch.Tensor]):
        projected = [proj(feat) for proj, feat in zip(self.projection, features)]
        
        stacked = torch.stack(projected, dim=0) 
        attended, _ = self.attention(stacked, stacked, stacked)
        combined = attended.mean(dim=0)  
        return self.layer_norm(combined)

class EnhancedProjection(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm2 = nn.LayerNorm(hidden_dim)     
        self.projection_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.final_layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        x = self.input_proj(x)
        x = self.layer_norm1(x)
        if len(x.shape) == 3:
            attn_out, _ = self.self_attention(x, x, x)
            x = self.layer_norm2(x + attn_out)
        for proj, norm in zip(self.projection_layers, self.layer_norms):
            residual = x
            x = proj(x)
            x = norm(x + residual)
        
        if len(x.shape) == 3:
            x = x.mean(dim=1) 
        x = self.output_proj(x)
        x = self.final_layer_norm(x)
        
        return x

class VideoEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.input_norm = nn.LayerNorm(input_dim)
        self.projection = EnhancedProjection(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=3 
        )
        
    def forward(self, x):
        x = self.input_norm(x)
        x = torch.clamp(x, min=-1.0, max=1.0)
        
        global_context = torch.mean(x, dim=1, keepdim=True)
        attention_weights = torch.matmul(x, global_context.transpose(1, 2))
        attention_weights = F.softmax(attention_weights, dim=1)
        diversity_penalty = torch.matmul(attention_weights, attention_weights.transpose(1, 2))
        attention_weights = attention_weights - 0.1 * diversity_penalty.mean(dim=2, keepdim=True)
        attention_weights = F.softmax(attention_weights, dim=1)
        x = torch.sum(x * attention_weights, dim=1)
        x = self.projection(x)
        return F.normalize(x, p=2, dim=-1)

class LanguageSpecificEncoder(nn.Module):
    def __init__(self, model_name: str, output_dim: int):
        super().__init__()
        if 'chinese' in model_name.lower():
            self.encoder = BertModel.from_pretrained('bert-base-chinese')
        else:
            self.encoder = XLMRobertaModel.from_pretrained(model_name)
            
        self.projection = EnhancedProjection(
            input_dim=self.encoder.config.hidden_size,
            hidden_dim=self.encoder.config.hidden_size,
            output_dim=output_dim,
            num_layers=2,  
            dropout=0.1
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.projection(cls_output)

class MultilingualDualEncoder(nn.Module):
    def __init__(self,
                 video_input_dim: int = 1024,
                 text_input_dim: int = 768,
                 output_dim: int = 256,
                 languages: List[str] = ['chinese', 'german']):
        if languages != ['chinese', 'german']:
            raise ValueError("This implementation only supports ['chinese', 'german'] as languages")
            
        super().__init__()
        self.languages = languages
        self.output_dim = output_dim
        self.video_encoder = VideoEncoder(video_input_dim, output_dim * 2, output_dim)
        self.text_encoders = nn.ModuleDict({
            'chinese': LanguageSpecificEncoder('bert-base-chinese', output_dim),
            'german': LanguageSpecificEncoder('xlm-roberta-base', output_dim)
        })
        
        self.video_projection = EnhancedProjection(output_dim, output_dim * 2, output_dim)
        self.text_projections = nn.ModuleDict({
            lang: EnhancedProjection(output_dim, output_dim * 2, output_dim)
            for lang in languages
        })
        
        self._init_weights()
        self.loss_fn = MultilingualContrastiveLoss(margin=0.2, temperature=0.05)
    
    def _init_weights(self):
        """Initialize weights with proper scaling."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.MultiheadAttention):
                nn.init.xavier_uniform_(m.in_proj_weight)
                if m.in_proj_bias is not None:
                    nn.init.zeros_(m.in_proj_bias)
                nn.init.xavier_uniform_(m.out_proj.weight)
                if m.out_proj.bias is not None:
                    nn.init.zeros_(m.out_proj.bias)
    
    def encode_video(self, video_features):
        """Encoding the video features into the shared embedding space."""
        video_emb = self.video_encoder(video_features)
        video_emb = self.video_projection(video_emb)
        video_emb = F.normalize(video_emb, p=2, dim=-1)
        return video_emb
    
    def encode_text(self, input_ids, attention_mask, language):
        """Encoding the texts into the shared embedding space."""
        text_emb = self.text_encoders[language](input_ids, attention_mask)
        text_emb = self.text_projections[language](text_emb)
        text_emb = F.normalize(text_emb, p=2, dim=-1)
        return text_emb
    
    def forward(self, video_features, text_inputs):
        video_emb = self.encode_video(video_features)
        text_embs = {}
        for lang in self.languages:
            if lang in text_inputs:
                text_embs[lang] = self.encode_text(
                    text_inputs[lang]['input_ids'],
                    text_inputs[lang]['attention_mask'],
                    lang
                )
        
        return {
            'video_embeddings': video_emb,
            'text_embeddings': text_embs
        } 