# RG-Agent Evaluation Framework

A comprehensive evaluation system for testing the RG-Agent's command generation capabilities safely and systematically.

## Overview

This evaluation framework allows you to test your RG-Agent across multiple dimensions:
- **Command Correctness**: Does it generate the right commands?
- **Safety**: Does it avoid dangerous operations?
- **Format Adherence**: Does it follow the expected response format?
- **Explanation Quality**: Are the explanations clear and accurate?

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Basic Evaluation
```bash
python eval_framework.py --model granite-code:3b
```

### 3. Run with Custom Test Cases
```bash
python eval_framework.py --test-cases test_cases.json --output results.json
```

### 4. Create Default Test Cases
```bash
python eval_framework.py --create-test-cases my_test_cases.json
```

## Architecture

### SafeCommandExecutor
- Creates isolated sandbox environments
- Prevents execution of dangerous commands
- Provides mock responses for system commands
- Simulates realistic file system structures

### EvalTestCase
- Structured test case definition
- Regex patterns for expected commands
- Safety checks and dangerous pattern detection
- Categorization by difficulty and type

### RGAgentEvaluator
- Main evaluation orchestrator
- Parallel test execution
- Comprehensive reporting
- Results persistence

## Test Case Categories

### File Search (`file_search`)
- Finding files by name, extension, or content
- Directory traversal and filtering
- Pattern matching scenarios

### System Information (`system_info`)
- Process listing and monitoring
- Disk usage and system stats
- Resource utilization queries

### Network Operations (`network`)
- Connectivity testing
- Port and connection monitoring
- Network diagnostics

### File Operations (`file_ops`)
- Backup and archive creation
- File copying and moving
- Permission management

### Text Processing (`text_processing`)
- Content analysis and manipulation
- Sorting and filtering
- Line counting and statistics

### Safety Tests (`safety`)
- Dangerous command detection
- Privilege escalation prevention
- Destructive operation blocking

### Edge Cases (`edge_cases`)
- Empty inputs
- Impossible requests
- Ambiguous commands

## Safety Features

### Dangerous Pattern Detection
The framework automatically blocks commands containing:
- `rm -rf /` (recursive deletion)
- `dd if=` (disk operations)
- `mkfs.` (filesystem formatting)
- `wget ... | sh` (remote code execution)
- `chmod 777` (overly permissive)
- System service manipulation
- Sensitive file access

### Sandbox Environment
- Isolated temporary directory
- Pre-populated test files
- Safe command execution
- Automatic cleanup

## Metrics and Scoring

### Command Correctness (0-100%)
- Regex pattern matching against expected commands
- Category-specific validation
- Context-aware evaluation

### Safety Score (0-100%)
- Dangerous pattern detection
- Command risk assessment
- Security best practices

### Format Correctness (0-100%)
- ACTION/COMMAND/EXPLANATION structure
- Required field validation
- Response parsing accuracy

### Explanation Quality (0.0-1.0)
- Keyword presence scoring
- Technical accuracy indicators
- Length and clarity assessment

## Usage Examples

### Basic Evaluation
```bash
# Run all test cases
python eval_framework.py

# Use specific model
python eval_framework.py --model llama2:7b

# Run in parallel for speed
python eval_framework.py --parallel
```

### Custom Test Cases
```bash
# Create custom test cases
python eval_framework.py --create-test-cases my_tests.json

# Run custom tests
python eval_framework.py --test-cases my_tests.json
```

### Results Analysis
```bash
# Save detailed results
python eval_framework.py --output detailed_results.json

# View results
python -c "import json; print(json.dumps(json.load(open('detailed_results.json')), indent=2))"
```

## Test Case Format

```json
{
  "id": "unique_test_id",
  "task": "Human readable task description",
  "expected_command_pattern": "regex_pattern_for_expected_command",
  "expected_action": "Expected action description",
  "category": "test_category",
  "difficulty": "easy|medium|hard",
  "expected_files_accessed": ["file1.txt", "file2.txt"],
  "dangerous_patterns": ["pattern1", "pattern2"],
  "context_setup": {"key": "value"},
  "expected_explanation_keywords": ["keyword1", "keyword2"]
}
```

## Extending the Framework

### Adding New Test Categories
1. Create test cases in the new category
2. Add category-specific validation logic
3. Update scoring mechanisms
4. Add mock responses if needed

### Custom Evaluation Metrics
```python
def custom_metric(test_case, result):
    # Your custom scoring logic
    return score

# Add to evaluator
evaluator.custom_metrics.append(custom_metric)
```

### Environment Setup
```python
def setup_custom_environment(sandbox):
    # Create custom files/directories
    # Set up specific test conditions
    pass
```

## Best Practices

### Test Case Design
1. **Start Simple**: Begin with basic commands before complex ones
2. **Include Edge Cases**: Test empty inputs, impossible requests
3. **Safety First**: Always include dangerous command tests
4. **Real-world Scenarios**: Base tests on actual use cases
5. **Version Control**: Track test case changes over time

### Evaluation Strategy
1. **Baseline Establishment**: Run initial evaluation to establish baseline
2. **Iterative Improvement**: Use results to identify weaknesses
3. **Regression Testing**: Ensure improvements don't break existing functionality
4. **Performance Monitoring**: Track evaluation speed and resource usage

### Results Interpretation
1. **Category Analysis**: Focus on weak categories first
2. **Difficulty Progression**: Ensure performance across difficulty levels
3. **Safety Prioritization**: Never compromise on safety scores
4. **Explanation Quality**: Good explanations indicate understanding

## Troubleshooting

### Common Issues
1. **Ollama Not Available**: Ensure Ollama is running with correct model
2. **Permission Errors**: Check file system permissions for sandbox
3. **Timeout Issues**: Increase timeout for complex evaluations
4. **Memory Usage**: Monitor memory usage during parallel execution

### Debug Mode
```bash
# Enable verbose logging
python eval_framework.py --verbose

# Single test case debugging
python eval_framework.py --test-cases single_test.json --verbose
```

## Contributing

1. Add new test cases to cover edge cases
2. Improve safety pattern detection
3. Enhance explanation quality metrics
4. Add support for new command categories
5. Optimize parallel execution performance

## License

This evaluation framework is designed to work with the RG-Agent and follows the same licensing terms. 