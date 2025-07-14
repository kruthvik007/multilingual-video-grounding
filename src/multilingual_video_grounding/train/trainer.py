import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from typing import Dict, List
import numpy as np

from ..models.dual_encoder import MultilingualDualEncoder
from ..utils.data_loader import create_dataloaders
from ..utils.loss import MultilingualContrastiveLoss

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
    loss_fn: nn.Module,
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
        with torch.amp.autocast('cuda'):
            outputs = model(video_features, text_inputs)
            temperatures = {
                f"{lang1}_{lang2}": torch.tensor(1.0, device=device)
                for lang1 in model.languages
                for lang2 in model.languages
                if lang1 != lang2
            }
            loss = loss_fn(outputs, temperatures)
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
    device: torch.device,
    loss_fn: nn.Module
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
            with torch.amp.autocast('cuda'):
                outputs = model(video_features, text_inputs)
                temperatures = {
                    f"{lang1}_{lang2}": torch.tensor(1.0, device=device)
                    for lang1 in model.languages
                    for lang2 in model.languages
                    if lang1 != lang2
                }
                loss = loss_fn(outputs, temperatures)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

def train_model(config: Dict):
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    model = MultilingualDualEncoder(
        video_input_dim=1024,
        text_input_dim=768,
        output_dim=config['output_dim'],
        languages=config['languages']
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    loss_fn = MultilingualContrastiveLoss(margin=0.5, temperature=1.0)
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
        train_loss = train_epoch(
            model, train_loader, optimizer, device, loss_fn, scaler
        )
        val_loss = evaluate(model, val_loader, device, loss_fn)
        scheduler.step()
        print("\nEpoch Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("\nEarly stopping triggered!")
            break
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("\nSaving best model...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }, 'best_model.pt')
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("\nSaving final model...")
    torch.save({
        'epoch': config['num_epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss
    }, 'final_model.pt')
    
    return model
