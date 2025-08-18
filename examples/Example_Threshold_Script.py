"""
Comprehensive Testing Dataset for Sentinel Hate Speech Detection
================================================================

This module creates realistic user profiles with different speech patterns
to test threshold behavior and aggregation performance.
To better review statistics, messages and reasoning run the file with a --review or -r flag
Current optimal settings are at a 5:1 at a 0.01 temperature, as 1:1 at any temperature either underflags or gives false flags.


poetry run python examples/Example_Threshold_Script.py [FLAGS]

User Types:
- Normal Speech Only (3 users)
- Hate Speech Focused (2 users) 
- Sexual Content Focused (2 users)
- Mixed Content (2 users)
- All Types Combined (1 user)
"""

from sentinel.sentinel_local_index import SentinelLocalIndex
import numpy as np
from typing import Dict, List, Tuple

def create_user_profiles() -> Dict[str, List[str]]:
    """Create 10 different user profiles with varying speech patterns."""
    
    # Normal speech examples
    normal_speech = [
        "Hey how are you doing today?",
        "What are you up to this weekend?",
        "I love this new video game",
        "The weather is really nice today",
        "Did you see that movie last night?",
        "I'm going to the store later",
        "Work was pretty busy today",
        "Let's grab lunch sometime",
        "That's a really cool picture",
        "I just finished reading a great book",
        "How was your vacation?",
        "I'm excited for the weekend",
        "Thanks for helping me out",
        "Have a great day!",
        "What's your favorite food?",
        "I'm learning to cook pasta",
        "The concert was amazing",
        "I need to go shopping",
        "My family is visiting next week",
        "That joke was hilarious"
    ]
    
    # Hate speech examples
    hate_speech = [
        "I hate all those people",
        "they don't belong here",
        "those immigrants are ruining everything",
        "whites are superior to everyone else", 
        "the holocaust never happened",
        "Hitler had the right idea",
        "Jews control all the banks",
        "Black people are all criminals",
        "Muslims are all terrorists",
        "gay people are disgusting",
        "women belong in the kitchen",
        "trans people are mentally ill",
        "minorities are destroying our country",
        "we need to get rid of them all",
        "violence is the only solution"
    ]
    
    # Sexual content examples
    sexual_content = [
        "want to see me naked?",
        "send me some hot pics",
        "let's hook up tonight",
        "I'm so horny right now",
        "what are you wearing?",
        "wanna have some fun?",
        "you look so sexy",
        "let's get physical",
        "I want to touch you",
        "meet me in private",
        "show me your body",
        "let's talk dirty",
        "I'm thinking about you",
        "we could be alone together",
        "send nudes"
    ]
    
    users = {}
    
    # Normal Speech Only Users (3 users)
    for i in range(1, 4):
        users[f"normal_user_{i}"] = np.random.choice(normal_speech, size=15, replace=False).tolist()
    
    # Hate Speech Focused Users (2 users)
    for i in range(1, 3):
        hate_msgs = np.random.choice(hate_speech, size=10, replace=False).tolist()
        normal_msgs = np.random.choice(normal_speech, size=5, replace=False).tolist()
        users[f"hate_user_{i}"] = hate_msgs + normal_msgs
        np.random.shuffle(users[f"hate_user_{i}"])
    
    # Sexual Content Focused Users (2 users)  
    for i in range(1, 3):
        sexual_msgs = np.random.choice(sexual_content, size=10, replace=False).tolist()
        normal_msgs = np.random.choice(normal_speech, size=5, replace=False).tolist()
        users[f"sexual_user_{i}"] = sexual_msgs + normal_msgs
        np.random.shuffle(users[f"sexual_user_{i}"])
    
    # Mixed Content Users (2 users)
    for i in range(1, 3):
        hate_msgs = np.random.choice(hate_speech, size=5, replace=False).tolist()
        sexual_msgs = np.random.choice(sexual_content, size=5, replace=False).tolist()
        normal_msgs = np.random.choice(normal_speech, size=5, replace=False).tolist()
        users[f"mixed_user_{i}"] = hate_msgs + sexual_msgs + normal_msgs
        np.random.shuffle(users[f"mixed_user_{i}"])
    
    # All Types Combined User (1 user)
    hate_msgs = np.random.choice(hate_speech, size=7, replace=False).tolist()
    sexual_msgs = np.random.choice(sexual_content, size=7, replace=False).tolist()
    normal_msgs = np.random.choice(normal_speech, size=6, replace=False).tolist()
    users["all_types_user"] = hate_msgs + sexual_msgs + normal_msgs
    np.random.shuffle(users["all_types_user"])
    
    return users

def test_thresholds_and_ratios(review_mode: bool = False):
    """Test different threshold and ratio combinations.
    
    Args:
        review_mode: If True, shows detailed per-user analysis and missed detections
    """
    
    print("🧪 COMPREHENSIVE SENTINEL TESTING")
    if review_mode:
        print("📋 REVIEW MODE: Detailed analysis enabled")
    print("=" * 50)
    
    # Load index with different ratios
    ratios_to_test = [10.0, 5.0, 1.0]
    # Temperature as defined in the model.
    thresholds_to_test = [0.0, 0.01, 0.05, 0.1] 
    
    users = create_user_profiles()
    
    for ratio in ratios_to_test:
        ratio_name = "Original" if ratio is None else f"{ratio}:1"
        print(f"\n📊 TESTING RATIO: {ratio_name}")
        print("-" * 30)
        
        # Load index with specific ratio
        index = SentinelLocalIndex.load(
            path="path/to/local/index",
            negative_to_positive_ratio=ratio
        )
        
        # Or load from S3
        
        # index = SentinelLocalIndex.load(
        #     path="s3://my-bucket/path/to/index",
        #     aws_access_key_id="YOUR_ACCESS_KEY_ID",  # Optional if using environment credentials
        #     aws_secret_access_key="YOUR_SECRET_ACCESS_KEY",  # Optional if using environment credentials
        #     negative_to_positive_ratio=ratio
        # )
        
        print(f"Loaded shapes: pos={index.positive_embeddings.shape[0]}, neg={index.negative_embeddings.shape[0]}")
        
        for threshold in thresholds_to_test:
            print(f"\n🎯 Threshold: {threshold}")
            
            results = {}
            
            # Test each user
            for user_name, messages in users.items():
                result = index.calculate_rare_class_affinity(
                    messages,
                    min_score_to_consider=threshold
                )
                
                # Calculate statistics
                positive_scores = [score for score in result.observation_scores.values() if score > 0]
                results[user_name] = {
                    'overall_score': result.rare_class_affinity_score,
                    'positive_count': len(positive_scores),
                    'max_score': max(result.observation_scores.values()) if result.observation_scores.values() else 0,
                    'avg_positive': np.mean(positive_scores) if positive_scores else 0
                }
            
            # Categorize and display results
            categories = {
                'Normal Users': [k for k in results.keys() if k.startswith('normal_')],
                'Hate Users': [k for k in results.keys() if k.startswith('hate_')], 
                'Sexual Users': [k for k in results.keys() if k.startswith('sexual_')],
                'Mixed Users': [k for k in results.keys() if k.startswith('mixed_')],
                'All Types': [k for k in results.keys() if k.startswith('all_types')]
            }
            
            for category, user_list in categories.items():
                if not user_list:
                    continue
                    
                scores = [results[user]['overall_score'] for user in user_list]
                detections = [results[user]['positive_count'] for user in user_list]
                
                print(f"  {category:12}: avg_score={np.mean(scores):.4f}, avg_detections={np.mean(detections):.1f}")
            
            # Summary statistics
            normal_scores = [results[u]['overall_score'] for u in categories['Normal Users']]
            problematic_scores = [results[u]['overall_score'] for u in 
                                categories['Hate Users'] + categories['Sexual Users'] + 
                                categories['Mixed Users'] + categories['All Types']]
            
            if normal_scores and problematic_scores:
                separation = np.mean(problematic_scores) - np.mean(normal_scores)
                print(f"  📈 Separation: {separation:.4f} (higher is better)")
            
            # Detailed review mode analysis
            if review_mode:
                print("\n🔍 DETAILED REVIEW:")
                
                # Analyze missed detections (problematic users with low scores)
                missed_detections = []
                false_positives = []
                
                for category, user_list in categories.items():
                    if not user_list:
                        continue
                        
                    print(f"\n  📊 {category} Analysis:")
                    
                    for user in user_list:
                        user_result = results[user]
                        score = user_result['overall_score']
                        detections = user_result['positive_count']
                        max_score = user_result['max_score']
                        
                        # Determine if this should be flagged based on category
                        should_be_flagged = not user.startswith('normal_')
                        is_flagged = score > 0.1  # Using 0.1 as detection threshold
                        
                        status = "✅" if (should_be_flagged == is_flagged) else "❌"
                        
                        print(f"    {status} {user:15}: score={score:.4f}, detections={detections:2d}, max={max_score:.4f}")
                        
                        # Track problematic cases
                        if should_be_flagged and not is_flagged:
                            missed_detections.append({
                                'user': user,
                                'category': category,
                                'score': score,
                                'detections': detections,
                                'max_score': max_score
                            })
                        elif not should_be_flagged and is_flagged:
                            false_positives.append({
                                'user': user,
                                'category': category,
                                'score': score,
                                'detections': detections,
                                'max_score': max_score
                            })
                
                # Summary of issues
                if missed_detections:
                    print(f"\n  ⚠️  MISSED DETECTIONS ({len(missed_detections)}):")
                    for miss in missed_detections:
                        print(f"    - {miss['user']} ({miss['category']}): score={miss['score']:.4f}")
                
                if false_positives:
                    print(f"\n  🚨 FALSE POSITIVES ({len(false_positives)}):")
                    for fp in false_positives:
                        print(f"    - {fp['user']} ({fp['category']}): score={fp['score']:.4f}")
                
                if not missed_detections and not false_positives:
                    print(f"\n  🎯 PERFECT CLASSIFICATION at threshold {threshold}")
                
                # Show sample messages for problematic cases
                if missed_detections and len(missed_detections) <= 2:
                    print(f"\n  📝 Sample messages from missed detections:")
                    for miss in missed_detections[:2]:
                        user_name = miss['user']
                        user_messages = users[user_name]
                        print(f"\n    {user_name} messages (showing first 5):")
                        for i, msg in enumerate(user_messages[:5]):
                            print(f"      {i+1}. \"{msg}\"")
                
                print(f"\n  📊 Classification Summary:")
                total_users = len([u for cat in categories.values() for u in cat])
                correct_classifications = total_users - len(missed_detections) - len(false_positives)
                accuracy = correct_classifications / total_users if total_users > 0 else 0
                print(f"    Accuracy: {accuracy:.2%} ({correct_classifications}/{total_users})")
                print(f"    Missed: {len(missed_detections)}, False Positives: {len(false_positives)}")

def main():
    """Run the comprehensive testing."""
    import sys
    
    # Check for review flag
    review_mode = '--review' in sys.argv or '-r' in sys.argv
    
    if review_mode:
        print("🔍 Review mode enabled - showing detailed analysis")
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    test_thresholds_and_ratios(review_mode=review_mode)
    
    print(f"\n✅ Testing complete! Check results above to determine optimal threshold and ratio.")
    if not review_mode:
        print("💡 Tip: Run with --review or -r flag for detailed per-user analysis")

if __name__ == "__main__":
    main()
