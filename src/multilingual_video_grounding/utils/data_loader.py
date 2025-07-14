import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, XLMRobertaTokenizer
from typing import Dict, List, Tuple, Union
import torch.nn.functional as F
from tqdm import tqdm

def collate_fn(batch):
    video_embeddings = torch.stack([item['video_embedding'] for item in batch])
    captions = {}
    for lang in ['chinese', 'german']:
        captions[lang] = {
            'input_ids': torch.stack([item['captions'][lang]['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['captions'][lang]['attention_mask'] for item in batch])
        }
    video_ids = [item['video_id'] for item in batch]
    return {
        'video_embedding': video_embeddings,
        'captions': captions,
        'video_id': video_ids
    }

class VideoCaptionDataset(Dataset):
    def __init__(self, 
                 json_path: str,
                 embeddings_dir: str,
                 tokenizers: Dict[str, Union[BertTokenizer, XLMRobertaTokenizer]],
                 max_length: int = 77,
                 max_captions: int = 5,
                 subset_size: float = 1.0):
        self.json_path = json_path
        self.embeddings_dir = embeddings_dir
        self.tokenizers = tokenizers
        self.max_length = max_length
        self.max_captions = max_captions
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        if subset_size < 1.0:
            items = list(self.data.items())
            subset_size = int(len(items) * subset_size)
            self.data = dict(items[:subset_size])
        self.valid_indices = []
        for video_id, video_data in self.data.items():
            embedding_path = os.path.join(embeddings_dir, f"{video_id}.npy")
            if os.path.exists(embedding_path):
                self.valid_indices.append(video_id)
        print(f"Loaded {len(self.valid_indices)} valid video-caption pairs")
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        video_id = self.valid_indices[idx]
        video_data = self.data[video_id]
        embedding_path = os.path.join(self.embeddings_dir, f"{video_id}.npy")
        video_embedding = torch.from_numpy(np.load(embedding_path)).float()
        if len(video_embedding.shape) == 1:
            video_embedding = video_embedding.unsqueeze(0)
        elif len(video_embedding.shape) == 3:
            video_embedding = video_embedding.mean(dim=1)
        captions = {}
        for lang in ['chinese', 'german']:
            if lang in video_data and video_data[lang]:
                num_captions = min(self.max_captions, len(video_data[lang]))
                captions_list = video_data[lang][:num_captions]
                tokens = [self.tokenizers[lang](
                    c, padding='max_length', max_length=self.max_length,
                    truncation=True, return_tensors='pt') for c in captions_list]
                input_ids = torch.stack([t['input_ids'].squeeze(0) for t in tokens])
                attention_mask = torch.stack([t['attention_mask'].squeeze(0) for t in tokens])
                captions[lang] = {
                    'input_ids': input_ids.float().mean(dim=0).long(),
                    'attention_mask': (attention_mask.float().mean(dim=0) > 0).long()
                }
            else:
                captions[lang] = {
                    'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                    'attention_mask': torch.zeros(self.max_length, dtype=torch.long)
                }
        return {
            'video_embedding': video_embedding,
            'captions': captions,
            'video_id': video_id
        }

def create_dataloaders(
    train_json_path: str,
    val_json_path: str,
    train_embeddings_dir: str,
    val_embeddings_dir: str,
    batch_size: int = 16,
    num_workers: int = 4,
    max_captions: int = 5,
    languages: List[str] = ['chinese', 'german'],
    subset_size: float = 1.0
) -> Tuple[DataLoader, DataLoader]:
    tokenizers = {
        'chinese': BertTokenizer.from_pretrained('bert-base-chinese'),
        'german': XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    }
    train_dataset = VideoCaptionDataset(
        train_json_path,
        train_embeddings_dir,
        tokenizers,
        max_captions=max_captions,
        subset_size=subset_size
    )
    val_dataset = VideoCaptionDataset(
        val_json_path,
        val_embeddings_dir,
        tokenizers,
        max_captions=max_captions,
        subset_size=subset_size
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return train_dataloader, val_dataloader
