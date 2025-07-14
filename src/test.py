import json
import random
from multilingual_video_grounding.eval import evaluate_model
import os

def load_paired_captions(json_path: str, num_samples: int = 5) -> list:
    """Loads a random Chinese captions and their corresponding German translations needed for testing."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)    
    paired_videos = []
    for video_id, video_data in data.items():
        if 'chinese' in video_data and 'german' in video_data:
            paired_videos.append({
                'video_id': video_id,
                'chinese': video_data['chinese'][0],  
                'german': video_data['german'][0]     
            })
    
    return random.sample(paired_videos, min(num_samples, len(paired_videos)))

def main():
    test_pairs = [
        {
            'chinese': '一个人拿着呼啦圈用手在头部转动随后掉入了腰部。',
            'german': 'Eine Dame dreht einen Hula Hope Ring auf ihren Wast.',
            'video_id': 'sSX1KaKyrag_000007_000017'  
        },
        {
            'chinese': '一个小朋友正在一个房子里面玩跑步机。',
            'german': 'Ein junges Mädchen versucht, ein Treadmill zu verwenden, das sehr dunkel ist.',
            'video_id': '0o-DvBPWSEI_000087_000097'  
        }
    ]
    
    print(f"Loaded {len(test_pairs)} test pairs\n")
    print("Selected test pairs:\n")
    for i, pair in enumerate(test_pairs, 1):
        print(f"Pair {i}:")
        print(f"Chinese: {pair['chinese']}")
        print(f"German (ground truth): {pair['german']}")
        print(f"Video ID: {pair['video_id']}\n")
        
    test_sentences = [pair['chinese'] for pair in test_pairs]
    results = evaluate_model(
        model_path='best_model.pt',
        test_sentences=test_sentences,
        test_pairs=test_pairs
    )
    print("Test Results:\n")
    for i, (sentence, matches) in enumerate(results.items()):
        print(f"Chinese sentence: {sentence}")
        print(f"Expected German: {test_pairs[i]['german']}")
        print("Model's top matches:")
        for score, video_id, caption in matches:
            print(f"Score: {score:.4f}")
            print(f"Video ID: {video_id}")
            print(f"German caption: {caption}")
            print("---")

if __name__ == "__main__":
    main() 