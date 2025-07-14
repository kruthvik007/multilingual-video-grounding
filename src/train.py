import torch
from torch.utils.data import DataLoader
from multilingual_video_grounding.models.dual_encoder_triplet import MultilingualTripletEncoder
from multilingual_video_grounding.data.dataset import VideoTextDataset, collate_fn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging
import random
from typing import Dict, List, Tuple
import os
from tqdm import tqdm
import time

def create_negative_batch(batch: Dict[str, torch.Tensor], dataset: VideoTextDataset) -> Dict[str, torch.Tensor]:
    """Creates negative samples by randomly sampling from different videos."""
    batch_size = batch['video_features'].size(0)
    all_indices = set(range(len(dataset)))
    batch_indices = set(batch['indices'].tolist())
    available_indices = list(all_indices - batch_indices)
    neg_indices = random.sample(available_indices, batch_size)
    neg_samples = [dataset[idx] for idx in neg_indices]
    neg_batch = collate_fn(neg_samples)
    
    return {
        'video_features': neg_batch['video_features'],
        'chinese_ids': neg_batch['chinese_ids'],
        'chinese_mask': neg_batch['chinese_mask'],
        'german_ids': neg_batch['german_ids'],
        'german_mask': neg_batch['german_mask']
    }

def train_epoch(model: MultilingualTripletEncoder,
                dataloader: DataLoader,
                optimizer: optim.Optimizer,
                device: torch.device,
                scaler: torch.cuda.amp.GradScaler) -> float:
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    pbar = tqdm(dataloader, desc="Training", unit="batch")
    start_time = time.time()
    
    for batch_idx, batch in enumerate(pbar):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        if batch_idx == 0:
            print("\nInput tensor shapes:")
            print(f"Video features: {batch['video_features'].shape}")
            print(f"Chinese IDs: {batch['chinese_ids'].shape}")
            print(f"Chinese mask: {batch['chinese_mask'].shape}")
            print(f"German IDs: {batch['german_ids'].shape}")
            print(f"German mask: {batch['german_mask'].shape}")
        
        neg_batch = create_negative_batch(batch, dataloader.dataset)
        neg_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in neg_batch.items()}
        
        if batch_idx == 0:
            print("\nNegative sample shapes:")
            print(f"Neg video features: {neg_batch['video_features'].shape}")
            print(f"Neg Chinese IDs: {neg_batch['chinese_ids'].shape}")
            print(f"Neg Chinese mask: {neg_batch['chinese_mask'].shape}")
            print(f"Neg German IDs: {neg_batch['german_ids'].shape}")
            print(f"Neg German mask: {neg_batch['german_mask'].shape}")
        
        try:
            full_batch = {
                'video_features': batch['video_features'],
                'chinese_ids': batch['chinese_ids'],
                'chinese_mask': batch['chinese_mask'],
                'german_ids': batch['german_ids'],
                'german_mask': batch['german_mask'],
                'neg_video_features': neg_batch['video_features'],
                'neg_chinese_ids': neg_batch['chinese_ids'],
                'neg_chinese_mask': neg_batch['chinese_mask'],
                'neg_german_ids': neg_batch['german_ids'],
                'neg_german_mask': neg_batch['german_mask']
            }
            
            with torch.amp.autocast('cuda'):
                outputs = model(full_batch)
                loss = outputs['loss']
            
            if batch_idx == 0:
                print("\nModel output shapes:")
                print(f"Video embeddings: {outputs['video_embeddings'].shape}")
                print(f"Chinese embeddings: {outputs['chinese_embeddings'].shape}")
                print(f"German embeddings: {outputs['german_embeddings'].shape}")
                print(f"Initial loss: {loss.item():.4f}")
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            elapsed = time.time() - start_time
            avg_time = elapsed / (batch_idx + 1)
            remaining = avg_time * (num_batches - batch_idx - 1) 
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_time': f'{avg_time:.2f}s/batch',
                'remaining': f'{remaining/60:.1f}min'
            })
                
        except RuntimeError as e:
            print(f"\nError in batch {batch_idx}:")
            print(str(e))
            print("\nBatch shapes:")
            for k, v in full_batch.items():
                if isinstance(v, torch.Tensor):
                    print(f"{k}: {v.shape}")
            raise e  
    return total_loss / num_batches

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = MultilingualTripletEncoder(
        video_input_dim=1024,
        hidden_dim=512,
        output_dim=256
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    train_dataset = VideoTextDataset(
        video_dir='Data/train_video_embeddings',
        annotation_file='Data/vatex_training_v1_german_chinese_english.json',
        subset_size=0.01
    ) 
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,  #
        shuffle=True,
        num_workers=12,  
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    for batch in train_loader:
        print("\nFirst batch shapes:")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: {v.shape}")
        break 
    print("\nStarting training...")
    num_epochs = 1
    best_loss = float('inf')
    scaler = torch.amp.GradScaler('cuda')
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, device, scaler)
        print(f"Training loss: {train_loss:.4f}")
        scheduler.step()
        
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss
            }, 'best_model.pt')
            print("Saved new best model")

if __name__ == "__main__":
    main() 