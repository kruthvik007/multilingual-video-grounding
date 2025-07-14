import torch
import torch.nn as nn
from typing import Tuple

class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 256):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)

class DualProjection(nn.Module):
    def __init__(self, 
                 video_input_dim: int,
                 text_input_dim: int,
                 output_dim: int = 256):

        super().__init__()
        self.video_projection = ProjectionHead(video_input_dim, output_dim)
        self.text_projection = ProjectionHead(text_input_dim, output_dim)
    
    def forward(self, 
                video_embeddings: torch.Tensor,
                text_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.video_projection(video_embeddings), self.text_projection(text_embeddings) 