"""
Comprehensive Testing Dataset for Sentinel Hate Speech Detection
================================================================

This module creates realistic user profiles with different speech patterns
to test threshold behavior and aggregation performance.
This will also show the time scale for each itteration to ensure efficient
output time and load times for better optimisation.

Current optimal settings are at a 5:1 at a 0.01 temperature, as 1:1 at any temperature either underflags or gives false flags.


poetry run python examples/Example_Threshold_Script.py [FLAGS]
  --review, -r          Show detailed per-user analysis and missed detections
  --results-only        Show only rare class affinity scores per user (compact output)

User Types:
- Normal Speech Only (3 users)
- Hate Speech Focused (2 users) 
- Sexual Content Focused (2 users)
- Mixed Content (2 users)
- All Types Combined (1 user)
"""

from sentinel.sentinel_local_index import SentinelLocalIndex
import numpy as np
import time
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
    
    """
    Requires an index with sexual positive examples, current database only focuses on hate-speech.
    As such information relating to sexual content has been removed from the codebase but can be added back by removing notes.
    
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
    """
    
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
#        sexual_msgs = np.random.choice(sexual_content, size=10, replace=False).tolist() # Requires an index with positive sexual content examples
        normal_msgs = np.random.choice(normal_speech, size=5, replace=False).tolist()
        users[f"sexual_user_{i}"] =   normal_msgs # + sexual_msgs
        np.random.shuffle(users[f"sexual_user_{i}"])
    
    # Mixed Content Users (2 users)
    for i in range(1, 3):
        hate_msgs = np.random.choice(hate_speech, size=5, replace=False).tolist() 
#        sexual_msgs = np.random.choice(sexual_content, size=5, replace=False).tolist() # Requires an index with positive sexual content examples
        normal_msgs = np.random.choice(normal_speech, size=5, replace=False).tolist()
        users[f"mixed_user_{i}"] = hate_msgs  + normal_msgs # + sexual_msgs
        np.random.shuffle(users[f"mixed_user_{i}"])
    
    # All Types Combined User (1 user)
    hate_msgs = np.random.choice(hate_speech, size=7, replace=False).tolist()
#    sexual_msgs = np.random.choice(sexual_content, size=7, replace=False).tolist() # Requires an index with positive sexual content examples
    normal_msgs = np.random.choice(normal_speech, size=6, replace=False).tolist()
    users["all_types_user"] = hate_msgs  + normal_msgs # + sexual_msgs
    np.random.shuffle(users["all_types_user"])
    
    return users

def test_thresholds_and_ratios(review_mode: bool = False, results_only: bool = False):
    """Test different threshold and ratio combinations.
    
    Args:
        review_mode: If True, shows detailed per-user analysis and missed detections
        results_only: If True, shows only the rare class affinity scores per user
    """
    
    overall_start_time = time.time()
    
    print("🧪 COMPREHENSIVE SENTINEL TESTING")
    if review_mode:
        print("📋 REVIEW MODE: Detailed analysis enabled")
    elif results_only:
        print("📊 RESULTS ONLY MODE: Showing rare class affinity scores per user")
    print("⏱️  PERFORMANCE METRICS: Timing data collection enabled")
    print("=" * 50)
    
    # Load index with different ratios
    ratios_to_test = [10.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    # Temperature as defined in the model.
    thresholds_to_test = [0.0, 0.01, 0.25, 0.05, 0.1] 
    
    users = create_user_profiles()
    
    # Performance tracking
    total_load_time = 0
    total_analysis_time = 0
    performance_metrics = []
    
    for ratio in ratios_to_test:
        ratio_name = "Original" if ratio is None else f"{ratio}:1"
        print(f"\n📊 TESTING RATIO: {ratio_name}")
        print("-" * 30)
        
        # Time data loading
        load_start = time.time()
        index = SentinelLocalIndex.load(
            path="path/to/local/index",
            negative_to_positive_ratio=ratio
        )
        load_time = time.time() - load_start
        total_load_time += load_time
        
        # Or load from S3
        
        # index = SentinelLocalIndex.load(
        #     path="s3://my-bucket/path/to/index",
        #     aws_access_key_id="YOUR_ACCESS_KEY_ID",  # Optional if using environment credentials
        #     aws_secret_access_key="YOUR_SECRET_ACCESS_KEY",  # Optional if using environment credentials
        #     negative_to_positive_ratio=ratio
        # )
        
        print(f"Loaded shapes: pos={index.positive_embeddings.shape[0]}, neg={index.negative_embeddings.shape[0]}")
        print(f"⏱️  Load time: {load_time:.3f}s")
        
        for threshold in thresholds_to_test:
            print(f"\n🎯 Threshold: {threshold}")
            
            # Time analysis phase
            analysis_start = time.time()
            
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
                    'avg_positive': np.mean(positive_scores) if positive_scores else 0,
                    'result_object': result  # Store the full result for results_only mode
                }
            
            analysis_time = time.time() - analysis_start
            total_analysis_time += analysis_time
            
            # Calculate comprehensive metrics
            normal_users = [k for k in results.keys() if k.startswith('normal_')]
            hate_users = [k for k in results.keys() if k.startswith('hate_')]
            sexual_users = [k for k in results.keys() if k.startswith('sexual_')]
            mixed_users = [k for k in results.keys() if k.startswith('mixed_')]
            all_types_users = [k for k in results.keys() if k.startswith('all_types')]
            
            # Message-level metrics - dynamically calculate based on actual user profiles
            total_normal_messages = 0
            total_hate_messages = 0
            total_sexual_messages = 0
            
            # Count actual messages for each user type
            for user_name, messages in users.items():
                if user_name.startswith('normal_'):
                    total_normal_messages += len(messages)
                elif user_name.startswith('hate_') or user_name.startswith('mixed_') or user_name.startswith('all_types'):
                    # Count hate messages in these users (excluding normal messages)
                    if user_name.startswith('hate_'):
                        total_hate_messages += 10  # 10 hate messages per hate user
                    elif user_name.startswith('mixed_'):
                        total_hate_messages += 5   # 5 hate messages per mixed user
                    elif user_name.startswith('all_types'):
                        total_hate_messages += 7   # 7 hate messages per all_types user
                elif user_name.startswith('sexual_'):
                    # Currently sexual users only have normal messages (sexual content commented out)
                    total_normal_messages += len(messages)
            
            total_problematic_messages = total_hate_messages + total_sexual_messages 
            
            # False positives (normal messages flagged as problematic)
            false_positive_messages = sum(results[user]['positive_count'] for user in normal_users)
            false_positive_rate_messages = false_positive_messages / total_normal_messages if total_normal_messages > 0 else 0
            
            # True positives (hate/mixed messages correctly flagged - focusing on hate detection)
            true_positive_messages = sum(results[user]['positive_count'] for user in hate_users + mixed_users + all_types_users)
            true_positive_rate = true_positive_messages / total_problematic_messages if total_problematic_messages > 0 else 0
            
            # User-level false positives (for comparison)
            false_positive_users = sum(1 for user in normal_users if results[user]['overall_score'] > 0.01)
            false_positive_rate_users = false_positive_users / len(normal_users) if normal_users else 0
            
            # Overall accuracy (correct classifications / total messages)
            total_messages = sum(len(messages) for messages in users.values())  # Dynamic total based on actual message counts
            correct_normal_messages = total_normal_messages - false_positive_messages
            correct_problematic_messages = true_positive_messages
            message_accuracy = (correct_normal_messages + correct_problematic_messages) / total_messages
            
            # Store comprehensive performance metrics
            performance_metrics.append({
                'ratio': ratio,
                'threshold': threshold,
                'load_time': load_time,
                'analysis_time': analysis_time,
                'false_positive_rate_messages': false_positive_rate_messages,
                'false_positive_rate_users': false_positive_rate_users,
                'false_positive_messages': false_positive_messages,
                'false_positive_users': false_positive_users,
                'true_positive_rate': true_positive_rate,
                'true_positive_messages': true_positive_messages,
                'message_accuracy': message_accuracy,
                'total_messages': total_messages,
                'total_normal_messages': total_normal_messages,
                'total_problematic_messages': total_problematic_messages
            })
            
            print(f"⏱️  Analysis: {analysis_time:.3f}s | FP Messages: {false_positive_messages}/{total_normal_messages} ({false_positive_rate_messages:.1%}) | TP Rate: {true_positive_rate:.1%} | Accuracy: {message_accuracy:.1%}")
            
            # Results-only mode: show just the scores per user
            if results_only:
                print(f"\n🎯 User Rare Class Affinity Scores (Threshold: {threshold}):")
                
                # Group users by category for organized display
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
                    print(f"\n  📊 {category}:")
                    for user in user_list:
                        score = results[user]['overall_score']
                        detections = results[user]['positive_count']
                        # Enhanced FP indicator with message context
                        if user.startswith('normal_') and detections > 0:
                            fp_indicator = f" ⚠️ FP ({detections}/15 msgs)"
                        else:
                            fp_indicator = ""
                        print(f"    {user:20}: score={score:.4f} ({detections} detections){fp_indicator}")
                
                continue  # Skip the normal summary output
            
            # Categorize and display results (normal mode)
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

    # Performance summary
    overall_time = time.time() - overall_start_time
    
    print(f"\n⏱️  PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"Total execution time: {overall_time:.3f}s")
    print(f"Total data load time: {total_load_time:.3f}s ({total_load_time/overall_time:.1%})")
    print(f"Total analysis time: {total_analysis_time:.3f}s ({total_analysis_time/overall_time:.1%})")
    
    # Best performance metrics
    if performance_metrics:
        print(f"\n📊 OPTIMIZATION METRICS:")
        
        # Find optimal configurations based on comprehensive metrics
        zero_fp_configs = [m for m in performance_metrics if m['false_positive_rate_messages'] == 0]
        if zero_fp_configs:
            # Among zero FP configs, find best true positive rate
            best_zero_fp = max(zero_fp_configs, key=lambda x: x['true_positive_rate'])
            print(f"Best zero false positive: {best_zero_fp['ratio']}:1 @ {best_zero_fp['threshold']} ({best_zero_fp['analysis_time']:.3f}s TP: {best_zero_fp['true_positive_rate']:.1%})")
        
        # Best overall accuracy
        best_accuracy = max(performance_metrics, key=lambda x: x['message_accuracy'])
        print(f"Best message accuracy: {best_accuracy['ratio']}:1 @ {best_accuracy['threshold']} ({best_accuracy['analysis_time']:.3f}s {best_accuracy['message_accuracy']:.1%} accuracy, {best_accuracy['false_positive_rate_messages']:.1%} FP)")
        
        # Fastest with reasonable accuracy (>90%)
        fast_accurate = [m for m in performance_metrics if m['message_accuracy'] > 0.9]
        if fast_accurate:
            fastest_accurate = min(fast_accurate, key=lambda x: x['analysis_time'])
            print(f"Fastest with >90% accuracy: {fastest_accurate['ratio']}:1 @ {fastest_accurate['threshold']} ({fastest_accurate['analysis_time']:.3f}s, {fastest_accurate['message_accuracy']:.1%} acc)")
        
        fastest_overall = min(performance_metrics, key=lambda x: x['analysis_time'])
        print(f"Fastest analysis overall: {fastest_overall['ratio']}:1 @ {fastest_overall['threshold']} ({fastest_overall['analysis_time']:.3f}s, {fastest_overall['message_accuracy']:.1%} acc, {fastest_overall['false_positive_rate_messages']:.1%} FP)")
        
        # Performance by ratio with message-level metrics
        print(f"\nMessage-level performance by ratio:")
        for ratio in ratios_to_test:
            ratio_metrics = [m for m in performance_metrics if m['ratio'] == ratio]
            avg_time = np.mean([m['analysis_time'] for m in ratio_metrics])
            avg_fp_rate = np.mean([m['false_positive_rate_messages'] for m in ratio_metrics])
            avg_tp_rate = np.mean([m['true_positive_rate'] for m in ratio_metrics])
            avg_accuracy = np.mean([m['message_accuracy'] for m in ratio_metrics])
            print(f"  {ratio}:1 ratio: {avg_time:.3f}s avg, {avg_fp_rate:.1%} FP rate, {avg_tp_rate:.1%} TP rate, {avg_accuracy:.1%} accuracy")

def main():
    """Run the comprehensive testing."""
    import sys
    
    # Check for flags
    review_mode = '--review' in sys.argv or '-r' in sys.argv
    results_only = '--results-only' in sys.argv or '--results' in sys.argv
    
    # Ensure mutually exclusive modes
    if review_mode and results_only:
        print("❌ Error: Cannot use both --review and --results-only flags simultaneously")
        sys.exit(1)
    
    if review_mode:
        print("🔍 Review mode enabled - showing detailed analysis")
    elif results_only:
        print("📊 Results-only mode enabled - showing rare class affinity scores per user")
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    test_thresholds_and_ratios(review_mode=review_mode, results_only=results_only)
    
    print(f"\n✅ Testing complete! Check results above to determine optimal threshold and ratio.")
    if not review_mode and not results_only:
        print("💡 Tip: Run with --review (-r) for detailed analysis or --results-only for user scores only")

if __name__ == "__main__":
    main()
