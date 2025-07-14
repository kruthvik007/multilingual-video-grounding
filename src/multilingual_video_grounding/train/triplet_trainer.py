import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from typing import Dict, List
import numpy as np

from ..models.dual_encoder_triplet import MultilingualTripletEncoder, TripletLoss
from ..utils.data_loader import create_dataloaders

class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        
    def __call__(self, val_loss: float):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        video_features = batch['video_embedding'].to(device)
        text_inputs = {
            lang: {
                'input_ids': batch['captions'][lang]['input_ids'].to(device),
                'attention_mask': batch['captions'][lang]['attention_mask'].to(device)
            } for lang in model.languages
        }
        video_ids = batch['video_id']
        with torch.amp.autocast('cuda'):
            outputs = model(video_features, text_inputs)
            video_emb = outputs['video_embeddings']
            chinese_emb = outputs['text_embeddings']['chinese']
            german_emb = outputs['text_embeddings']['german']
            loss = model.loss_fn(video_emb, chinese_emb, german_emb, video_ids)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches

def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device
) -> float:
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            video_features = batch['video_embedding'].to(device)
            text_inputs = {
                lang: {
                    'input_ids': batch['captions'][lang]['input_ids'].to(device),
                    'attention_mask': batch['captions'][lang]['attention_mask'].to(device)
                } for lang in model.languages
            }
            video_ids = batch['video_id']
            with torch.amp.autocast('cuda'):
                outputs = model(video_features, text_inputs)
                video_emb = outputs['video_embeddings']
                chinese_emb = outputs['text_embeddings']['chinese']
                german_emb = outputs['text_embeddings']['german']
                loss = model.loss_fn(video_emb, chinese_emb, german_emb, video_ids)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

def train_model(config: Dict):
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    print("\nInitializing model...")
    model = MultilingualTripletEncoder(
        video_input_dim=1024,
        text_input_dim=768,
        output_dim=config['output_dim'],
        languages=config['languages']
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    train_loader, val_loader = create_dataloaders(
        train_json_path=config['train_json_path'],
        val_json_path=config['val_json_path'],
        train_embeddings_dir=config['train_embeddings_dir'],
        val_embeddings_dir=config['val_embeddings_dir'],
        batch_size=config['batch_size'],
        num_workers=4,
        languages=config['languages'],
        subset_size=config['subset_size']
    )
    print(f"Training samples: {len(train_loader.dataset):,}")
    print(f"Validation samples: {len(val_loader.dataset):,}")
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config['learning_rate'])
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs'] * len(train_loader),
        eta_min=config['learning_rate'] * 0.1
    )
    scaler = torch.amp.GradScaler('cuda', enabled=config['fp16'])
    print(f"Mixed precision training: {'enabled' if config['fp16'] else 'disabled'}")
    early_stopping = EarlyStopping(
        patience=config['early_stopping_patience'],
        min_delta=config['min_delta']
    )
    best_val_loss = float('inf')
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        print("-" * 50)
        train_loss = train_epoch(model, train_loader, optimizer, device, scaler)
        print(f"Training Loss: {train_loss:.4f}")
        val_loss = evaluate(model, val_loader, device)
        print(f"Validation Loss: {val_loss:.4f}")
        scheduler.step()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }, 'best_model.pt')
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    print("\n=== Training Complete ===")
    print(f"Best validation loss: {best_val_loss:.4f}")
    return model
