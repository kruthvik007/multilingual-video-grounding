import torch
import json
import os
import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizer
import torch.nn.functional as F
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from collections import defaultdict

from ..models.dual_encoder_triplet import MultilingualTripletEncoder

def load_model(model_path: str = 'best_model.pt', device: torch.device = None) -> MultilingualTripletEncoder:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = MultilingualTripletEncoder(
        video_input_dim=1024,
        hidden_dim=512,
        output_dim=256,
        languages=['chinese', 'german']
    ).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    print("\nLoaded checkpoint keys:", checkpoint.keys())
    model_state_dict = model.state_dict()
    for key, value in checkpoint['model_state_dict'].items():
        if key in model_state_dict:
            model_state_dict[key] = value
    model.load_state_dict(model_state_dict, strict=False)
    model.eval()
    return model

def process_video_embedding(
    model: MultilingualTripletEncoder,
    video_emb: np.ndarray,
    device: torch.device
) -> torch.Tensor:
    video_emb = torch.from_numpy(video_emb).float()
    video_emb = (video_emb - video_emb.mean()) / (video_emb.std() + 1e-8)
    video_emb = torch.clamp(video_emb, min=-1.0, max=1.0)
    video_emb = video_emb.to(device)
    video_emb = video_emb.squeeze(0)
    with torch.no_grad():
        video_emb = model.encode_video(video_emb.unsqueeze(0))
        video_emb = F.normalize(video_emb, p=2, dim=-1)
        return video_emb

def process_text(
    model: MultilingualTripletEncoder,
    text: str,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    language: str
) -> torch.Tensor:
    inputs = tokenizer(
        text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        text_emb = model.encode_text(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            language=language
        )
        text_emb = F.normalize(text_emb, p=2, dim=-1)
    return text_emb

def compute_similarity(text_emb: torch.Tensor, video_emb: torch.Tensor, temperature: float = 0.05) -> float:
    similarity = torch.sum(text_emb * video_emb)
    similarity = torch.exp(similarity / temperature)
    return similarity.item()

def get_diverse_matches(similarities: List[Tuple[float, str, str]], top_k: int = 3, diversity_weight: float = 0.7) -> List[Tuple[float, str, str]]:
    if len(similarities) <= top_k:
        return similarities
    similarities = sorted(similarities, key=lambda x: x[0], reverse=True)
    selected = [(similarities[0][0], similarities[0][1], similarities[0][2])]
    candidates = similarities[1:]
    while len(selected) < top_k and candidates:
        max_score = float('-inf')
        best_item = None
        best_idx = None
        for i, (score, vid_id, caption) in enumerate(candidates):
            penalty = sum(s[0] for s in selected) / len(selected)
            mmr_score = diversity_weight * score - (1 - diversity_weight) * penalty
            if mmr_score > max_score:
                max_score = mmr_score
                best_item = (score, vid_id, caption)
                best_idx = i
        if best_item is None:
            break
        selected.append(best_item)
        candidates.pop(best_idx)
    return selected

def visualize_embeddings(embeddings: np.ndarray, labels: List[str], title: str, save_path: str):
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    unique_labels = list(set(labels))
    palette = sns.color_palette("husl", len(unique_labels))
    color_dict = {label: palette[i] for i, label in enumerate(unique_labels)}
    plt.figure(figsize=(12, 8))
    for label in unique_labels:
        mask = [l == label for l in labels]
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                   c=[color_dict[label]], label=label, alpha=0.6)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def analyze_embeddings(
    model: MultilingualTripletEncoder,
    val_json_path: str,
    video_embedding_dir: str,
    device: torch.device,
    save_dir: str = 'visualizations'
):
    os.makedirs(save_dir, exist_ok=True)
    with open(val_json_path, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    tokenizers = {
        'chinese': AutoTokenizer.from_pretrained('bert-base-chinese'),
        'german': AutoTokenizer.from_pretrained('xlm-roberta-base')
    }
    video_embeddings = []
    text_embeddings = []
    video_labels = []
    text_labels = []
    for video_id, video_data in val_data.items():
        if 'chinese' not in video_data or 'german' not in video_data:
            continue
        embedding_path = os.path.join(video_embedding_dir, f"{video_id}.npy")
        if not os.path.exists(embedding_path):
            continue
        try:
            video_emb = np.load(embedding_path)
            video_emb = torch.from_numpy(video_emb).float()
            video_emb = (video_emb - video_emb.mean()) / (video_emb.std() + 1e-8)
            video_emb = torch.clamp(video_emb, min=-1.0, max=1.0)
            video_emb = process_video_embedding(model, video_emb.numpy(), device)
            video_embeddings.append(video_emb.cpu().numpy())
            video_labels.append(f"Video_{video_id}")
            for lang in ['chinese', 'german']:
                caption = video_data[lang][0]
                text_emb = process_text(model, caption, tokenizers[lang], device, lang)
                text_embeddings.append(text_emb.cpu().numpy())
                text_labels.append(f"{lang.capitalize()}_{video_id}")
        except Exception as e:
            print(f"Error processing video {video_id}: {str(e)}")
            continue
    if not video_embeddings:
        print("Warning: No video embeddings were successfully processed!")
        return
    video_embeddings = np.vstack(video_embeddings)
    text_embeddings = np.vstack(text_embeddings)
    print(f"\nSuccessfully processed {len(video_embeddings)} video embeddings")
    print(f"Successfully processed {len(text_embeddings)} text embeddings")
    try:
        visualize_embeddings(
            video_embeddings,
            video_labels,
            "t-SNE Visualization of Video Embeddings",
            os.path.join(save_dir, "video_embeddings_tsne.png")
        )
        print("Generated video embeddings t-SNE plot")
    except Exception as e:
        print(f"Error generating video embeddings t-SNE plot: {str(e)}")
    try:
        visualize_embeddings(
            text_embeddings,
            text_labels,
            "t-SNE Visualization of Text Embeddings",
            os.path.join(save_dir, "text_embeddings_tsne.png")
        )
        print("Generated text embeddings t-SNE plot")
    except Exception as e:
        print(f"Error generating text embeddings t-SNE plot: {str(e)}")
    try:
        all_embeddings = np.vstack([video_embeddings, text_embeddings])
        all_labels = video_labels + text_labels
        visualize_embeddings(
            all_embeddings,
            all_labels,
            "t-SNE Visualization of Combined Embeddings",
            os.path.join(save_dir, "combined_embeddings_tsne.png")
        )
        print("Generated combined embeddings t-SNE plot")
    except Exception as e:
        print(f"Error generating combined embeddings t-SNE plot: {str(e)}")

def evaluate_model(
    model_path: str = 'best_model.pt',
    val_json_path: str = 'Data/vatex_validation_v1_german_chinese_english.json',
    video_embedding_dir: str = 'Data/val_video_embeddings',
    test_sentences: List[str] = None,
    test_pairs: List[Dict[str, str]] = None,
    temperature: float = 1.0,
    diversity_weight: float = 0.5
) -> Dict[str, List[Tuple[float, str, str]]]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = load_model(model_path, device)
    tokenizers = {
        'chinese': AutoTokenizer.from_pretrained('bert-base-chinese'),
        'german': AutoTokenizer.from_pretrained('xlm-roberta-base')
    }
    video_captions = {}
    if test_pairs:
        for pair in test_pairs:
            if 'video_id' in pair and 'german' in pair:
                video_captions[pair['video_id']] = pair['german']
    video_embeddings = {}
    print("\nProcessing video embeddings...")
    print(f"Looking for video embeddings in: {os.path.abspath(video_embedding_dir)}")
    if not os.path.exists(video_embedding_dir):
        print(f"Error: Video embedding directory not found at {video_embedding_dir}")
        return {}
    valid_video_ids = set(video_captions.keys())
    print(f"Found {len(valid_video_ids)} valid video IDs from test pairs")
    for video_file in os.listdir(video_embedding_dir):
        if video_file.endswith('.npy'):
            video_id = video_file.replace('.npy', '')
            if video_id in valid_video_ids:
                video_path = os.path.join(video_embedding_dir, video_file)
                print(f"Loading video embedding from: {video_path}")
                try:
                    video_emb = np.load(video_path)
                    video_emb = torch.from_numpy(video_emb).float()
                    video_emb = (video_emb - video_emb.mean()) / (video_emb.std() + 1e-8)
                    video_emb = torch.clamp(video_emb, min=-1.0, max=1.0)
                    video_emb = process_video_embedding(model, video_emb.numpy(), device)
                    video_embeddings[video_id] = video_emb
                    print(f"Successfully processed video {video_id}")
                except Exception as e:
                    print(f"Error loading video embedding {video_id}: {str(e)}")
                    continue
    print(f"\nProcessed {len(video_embeddings)} video embeddings\n")
    if len(video_embeddings) == 0:
        print("Warning: No video embeddings were processed!")
        print("Valid video IDs from test pairs:", len(valid_video_ids))
        print("Sample video IDs from test pairs:", list(valid_video_ids)[:5])
        print("Sample video files:", [f for f in os.listdir(video_embedding_dir)[:5] if f.endswith('.npy')])
        return {}
    print("\nGenerating visualizations...")
    analyze_embeddings(
        model=model,
        val_json_path=val_json_path,
        video_embedding_dir=video_embedding_dir,
        device=device,
        save_dir='visualizations'
    )
    print("Visualizations generated in 'visualizations' directory")
    results = {}
    for sentence in test_sentences:
        print(f"\nChinese input: {sentence}")
        text_emb = process_text(model, sentence, tokenizers['chinese'], device, 'chinese')
        similarities = []
        for video_id, video_emb in video_embeddings.items():
            similarity = compute_similarity(text_emb, video_emb, temperature)
            similarities.append((similarity, video_id, video_captions[video_id]))
        top_matches = get_diverse_matches(similarities, top_k=3, diversity_weight=diversity_weight)
        results[sentence] = top_matches
        print("\nTop 3 matches:")
        for i, (score, video_id, caption) in enumerate(top_matches, 1):
            print(f"\n{i}. German translation: {caption}")
            print(f"   Similarity score: {score:.4f}")
            print(f"   Video ID: {video_id}")
    return results
