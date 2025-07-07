#!/usr/bin/env python3
"""
Example script for running RG-Agent evaluations
Demonstrates different evaluation scenarios and configurations
"""

import asyncio
import json
import sys
from eval_framework import RGAgentEvaluator

async def run_basic_evaluation():
    """Run a basic evaluation with default test cases"""
    print("🚀 Running Basic Evaluation")
    print("=" * 50)
    
    evaluator = RGAgentEvaluator(model="granite-code:3b")
    
    try:
        # Load default test cases
        evaluator.load_test_cases()
        print(f"📝 Loaded {len(evaluator.test_cases)} test cases")
        
        # Run evaluation
        report = await evaluator.run_evaluation(parallel=True)
        
        # Print report
        evaluator.print_report(report)
        
        # Save results
        evaluator.save_results("basic_evaluation_results.json")
        print("\n💾 Results saved to: basic_evaluation_results.json")
        
    finally:
        evaluator.cleanup()

async def run_safety_focused_evaluation():
    """Run evaluation focusing on safety test cases"""
    print("\n🔒 Running Safety-Focused Evaluation")
    print("=" * 50)
    
    evaluator = RGAgentEvaluator(model="granite-code:3b")
    
    try:
        # Load all test cases
        evaluator.load_test_cases()
        
        # Filter for safety test cases
        safety_cases = [tc for tc in evaluator.test_cases if tc.category == "safety"]
        evaluator.test_cases = safety_cases
        
        print(f"🔍 Running {len(safety_cases)} safety test cases")
        
        # Run evaluation
        report = await evaluator.run_evaluation(parallel=True)
        
        # Print report
        evaluator.print_report(report)
        
        # Check if any safety tests failed
        failed_safety = [case for case in report['failed_cases'] if 'safety_failed' in case.get('issues', [])]
        if failed_safety:
            print("\n❌ CRITICAL: Safety tests failed!")
            for case in failed_safety:
                print(f"  - {case['id']}: {case['task']}")
                print(f"    Command: {case['command']}")
        else:
            print("\n✅ All safety tests passed!")
            
    finally:
        evaluator.cleanup()

async def run_category_comparison():
    """Run evaluation and compare performance across categories"""
    print("\n📊 Running Category Comparison")
    print("=" * 50)
    
    evaluator = RGAgentEvaluator(model="granite-code:3b")
    
    try:
        # Load test cases
        evaluator.load_test_cases()
        
        # Run evaluation
        report = await evaluator.run_evaluation(parallel=True)
        
        # Analyze category performance
        print("\n📈 Category Performance Analysis:")
        categories = report['category_breakdown']
        
        for category, stats in sorted(categories.items(), key=lambda x: x[1]['correct']/x[1]['total'], reverse=True):
            correct_rate = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            safe_rate = stats['safe'] / stats['total'] if stats['total'] > 0 else 0
            
            print(f"\n{category.upper()}:")
            print(f"  ✅ Correctness: {correct_rate:.1%} ({stats['correct']}/{stats['total']})")
            print(f"  🔒 Safety: {safe_rate:.1%} ({stats['safe']}/{stats['total']})")
            
            if correct_rate < 0.7:
                print(f"  ⚠️  Needs improvement: Low correctness rate")
            if safe_rate < 1.0:
                print(f"  🚨 CRITICAL: Safety issues detected")
                
    finally:
        evaluator.cleanup()

async def run_progressive_difficulty():
    """Run evaluation by difficulty level to show progression"""
    print("\n📈 Running Progressive Difficulty Evaluation")
    print("=" * 50)
    
    evaluator = RGAgentEvaluator(model="granite-code:3b")
    
    try:
        # Load test cases
        evaluator.load_test_cases()
        
        # Run by difficulty level
        for difficulty in ['easy', 'medium', 'hard']:
            print(f"\n🎯 Testing {difficulty.upper()} difficulty")
            print("-" * 30)
            
            # Filter by difficulty
            difficulty_cases = [tc for tc in evaluator.test_cases if tc.difficulty == difficulty]
            evaluator.test_cases = difficulty_cases
            
            if not difficulty_cases:
                print(f"No {difficulty} test cases found")
                continue
            
            # Run evaluation
            report = await evaluator.run_evaluation(parallel=True)
            
            # Print summary
            summary = report['summary']
            print(f"Overall Score: {summary['overall_score']:.1%}")
            print(f"Command Correctness: {summary['command_correctness']:.1%}")
            print(f"Safety Score: {summary['safety_score']:.1%}")
            
        # Reload all test cases for final cleanup
        evaluator.load_test_cases()
        
    finally:
        evaluator.cleanup()

async def create_custom_test_cases():
    """Create and test custom test cases"""
    print("\n🛠️ Creating Custom Test Cases")
    print("=" * 50)
    
    # Define custom test cases
    custom_cases = [
        {
            "id": "custom_001",
            "task": "find all log files modified in the last hour",
            "expected_command_pattern": r"find\s+.*-name\s+[\"']?\*\.log[\"']?.*-mtime\s+-1",
            "expected_action": "Find recent log files",
            "category": "custom",
            "difficulty": "medium",
            "expected_files_accessed": None,
            "dangerous_patterns": None,
            "context_setup": None,
            "expected_explanation_keywords": ["find", "log", "mtime", "modified"]
        },
        {
            "id": "custom_002", 
            "task": "show the top 10 largest files in the current directory",
            "expected_command_pattern": r"du\s+.*\|\s*sort\s+.*\|\s*head\s+-10",
            "expected_action": "Find largest files",
            "category": "custom",
            "difficulty": "medium",
            "expected_files_accessed": None,
            "dangerous_patterns": None,
            "context_setup": None,
            "expected_explanation_keywords": ["du", "sort", "head", "largest"]
        }
    ]
    
    # Save custom test cases
    with open("custom_test_cases.json", "w") as f:
        json.dump(custom_cases, f, indent=2)
    
    print("📝 Created custom test cases: custom_test_cases.json")
    
    # Run evaluation on custom cases
    evaluator = RGAgentEvaluator(model="granite-code:3b")
    
    try:
        evaluator.load_test_cases("custom_test_cases.json")
        
        print(f"🧪 Testing {len(evaluator.test_cases)} custom test cases")
        
        # Run evaluation
        report = await evaluator.run_evaluation(parallel=True)
        
        # Print report
        evaluator.print_report(report)
        
    finally:
        evaluator.cleanup()

async def main():
    """Main function to run all example evaluations"""
    print("🧪 RG-Agent Evaluation Examples")
    print("=" * 60)
    
    # Check if we should run specific evaluation
    if len(sys.argv) > 1:
        eval_type = sys.argv[1].lower()
        
        if eval_type == "basic":
            await run_basic_evaluation()
        elif eval_type == "safety":
            await run_safety_focused_evaluation()
        elif eval_type == "categories":
            await run_category_comparison()
        elif eval_type == "difficulty":
            await run_progressive_difficulty()
        elif eval_type == "custom":
            await create_custom_test_cases()
        else:
            print(f"Unknown evaluation type: {eval_type}")
            print("Available types: basic, safety, categories, difficulty, custom")
    else:
        # Run all evaluations
        await run_basic_evaluation()
        await run_safety_focused_evaluation()
        await run_category_comparison()
        await run_progressive_difficulty()
        await create_custom_test_cases()

if __name__ == "__main__":
    print("Usage:")
    print("  python run_example_eval.py          # Run all examples")
    print("  python run_example_eval.py basic    # Run basic evaluation")
    print("  python run_example_eval.py safety   # Run safety-focused evaluation")
    print("  python run_example_eval.py categories # Run category comparison")
    print("  python run_example_eval.py difficulty # Run progressive difficulty")
    print("  python run_example_eval.py custom   # Create and test custom cases")
    print()
    
    asyncio.run(main()) 