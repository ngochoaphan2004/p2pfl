#!/usr/bin/env python3
"""
Script to filter and normalize P2PFL logs for convergence analysis.
Extracts round and metric information for plotting convergence curves.
"""

import re
import sys
import csv
import json
import ast
from typing import Dict, List, Tuple, Optional


def parse_log_line(line: str) -> Optional[Dict]:
    """
    Parse a log line to extract round and metric information.

    Args:
        line: A single log line

    Returns:
        Dictionary with round, metric_name, metric_value, and scope if found, None otherwise
    """
    # Extract round information
    round_match = re.search(r'round (\d+)', line)
    global_round_match = re.search(r'Round (\d+)', line)  # For capitalized "Round"

    round_num = None
    if round_match:
        round_num = int(round_match.group(1))
    elif global_round_match:
        round_num = int(global_round_match.group(1))

    # Extract metric information (accuracy, loss, etc.)
    # Pattern: accuracy, test_metric, loss, test_loss, train_loss, etc.
    metric_patterns = [
        r"accuracy.*?(\d+\.?\d*)",
        r"test_metric.*?(\d+\.?\d*)",
        r"loss.*?(\d+\.?\d*)",
        r"test_loss.*?(\d+\.?\d*)",
        r"train_loss.*?(\d+\.?\d*)"
    ]

    for pattern in metric_patterns:
        metric_match = re.search(pattern, line, re.IGNORECASE)
        if metric_match:
            metric_value = float(metric_match.group(1))

            # Determine metric name based on the pattern that matched
            if "accuracy" in pattern.lower():
                metric_name = "accuracy"
            elif "test_metric" in pattern.lower():
                metric_name = "test_metric"
            elif "train_loss" in pattern.lower():
                metric_name = "train_loss"
            elif "test_loss" in pattern.lower():
                metric_name = "test_loss"
            elif "loss" in pattern.lower():
                metric_name = "loss"
            else:
                metric_name = "unknown"

            return {
                "round": round_num,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "scope": "global"
            }

    # Look for evaluation results in the format like "Results: {accuracy: 0.8, loss: 0.3}"
    eval_match = re.search(r'Results: (.+)', line)
    if eval_match and round_num is not None:
        eval_str = eval_match.group(1)
        # Try to parse results dict - handle newlines in keys and values
        try:
            # The eval_str is a representation of a Python dict, not JSON
            # It may contain newlines in keys and values like: {'\ntest_loss': 14.449026107788086, 'test_acc\n': 0.13750000298023224, ...}

            # First, try to safely evaluate it as a Python literal
            try:
                # Use ast.literal_eval to safely parse the Python dict representation
                eval_dict = ast.literal_eval(eval_str)

                for metric_name, value in eval_dict.items():
                    if isinstance(value, (int, float)):
                        # Clean up metric name by removing newlines
                        clean_metric_name = str(metric_name).replace('\n', '').strip()
                        return {
                            "round": round_num,
                            "metric_name": clean_metric_name,
                            "metric_value": float(value),
                            "scope": "global"
                        }
            except (ValueError, SyntaxError):
                # If ast.literal_eval fails, try alternative parsing
                # Replace single quotes with double quotes to make it JSON-like and handle newlines
                cleaned_str = eval_str.strip()

                # Replace newlines in the string representation
                cleaned_str = cleaned_str.replace('\\n', '_newline')
                cleaned_str = cleaned_str.replace("'", '"')

                # Handle extra commas that might be generated
                cleaned_str = re.sub(r',\s*}', '}', cleaned_str)
                cleaned_str = re.sub(r',\s*]', ']', cleaned_str)

                eval_dict = json.loads(cleaned_str)

                for metric_name, value in eval_dict.items():
                    if isinstance(value, (int, float)):
                        # Clean up metric name by removing added "_newline" suffixes
                        clean_metric_name = metric_name.replace('_newline', '').replace('\\n', '').strip()
                        return {
                            "round": round_num,
                            "metric_name": clean_metric_name,
                            "metric_value": float(value),
                            "scope": "global"
                        }
        except json.JSONDecodeError:
            # Try alternative parsing approach - extract key-value pairs manually
            # The Results string has the format like: {'\ntest_loss': 14.449026107788086, 'test_acc\n': 0.13750000298023224, ...}
            try:
                # Pattern to match 'key': value pairs in the results, allowing for newlines in keys
                # This handles patterns like '\ntest_loss': value or 'test_acc\n': value
                # First remove the outer braces
                inner_content = eval_str.strip().strip('{}')

                # Split by comma, but be careful with nested structures
                # Match key-value pairs: 'key': value
                # Pattern matches: 'key': value where key may contain newlines
                kv_pairs = re.findall(r"'([^']*?)'\s*:\s*([0-9.]+)", inner_content)

                for key, value in kv_pairs:
                    # Clean the key by removing newlines from the key name
                    clean_key = key.replace('\n', '').strip()
                    if clean_key:  # Only if there's a clean key
                        return {
                            "round": round_num,
                            "metric_name": clean_key,
                            "metric_value": float(value),
                            "scope": "global"
                        }
            except:
                pass  # If alternative parsing fails, continue

            # If we can't parse the results, skip this line
            pass

    return None


def filter_and_normalize_logs(input_file: str, output_format: str = "csv") -> Tuple[List[str], List[Dict]]:
    """
    Filter and normalize logs to extract convergence data.

    Args:
        input_file: Path to the log file to process
        output_format: Format for output ('csv' or 'jsonl')

    Returns:
        Tuple of (filtered_lines, normalized_data)
    """
    filtered_lines = []
    normalized_data = []

    # Track current round as we parse the file
    current_round = None

    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()

            # Skip if line is empty
            if not line:
                continue

            # First, check if this line indicates a round change
            round_match = re.search(r'Round (\d+) of \d+', line)
            if round_match:
                current_round = int(round_match.group(1))

            # Check if line contains relevant convergence information
            if contains_relevant_info(line):
                # Parse the line for round and metrics
                parsed_data = parse_log_line_with_round(line, current_round)
                if parsed_data:
                    normalized_data.append(parsed_data)
                    if output_format == "csv":
                        filtered_lines.append(line)
                    else:  # JSONL
                        filtered_lines.append(json.dumps(parsed_data))
                else:
                    # Add as comment since it contains relevant info but not structured data
                    filtered_lines.append(f"# {line}")
            else:
                # Add as comment since it's not relevant for convergence analysis
                filtered_lines.append(f"# {line}")

    return filtered_lines, normalized_data


def parse_log_line_with_round(line: str, current_round: Optional[int]) -> Optional[Dict]:
    """
    Parse a log line to extract round and metric information, using the current round context.

    Args:
        line: A single log line
        current_round: The current round number based on context

    Returns:
        Dictionary with round, metric_name, metric_value, and scope if found, None otherwise
    """
    # Extract round information from the line itself first
    round_match = re.search(r'round (\d+)', line)
    global_round_match = re.search(r'Round (\d+)', line)  # For capitalized "Round"

    round_num = None
    if round_match:
        round_num = int(round_match.group(1))
    elif global_round_match:
        round_num = int(global_round_match.group(1))
    else:
        # Use the context round if no round is specified in the line
        round_num = current_round

    # Extract metric information (accuracy, loss, etc.)
    # Pattern: accuracy, test_metric, loss, test_loss, train_loss, etc.
    metric_patterns = [
        r"accuracy.*?(\d+\.?\d*)",
        r"test_metric.*?(\d+\.?\d*)",
        r"loss.*?(\d+\.?\d*)",
        r"test_loss.*?(\d+\.?\d*)",
        r"train_loss.*?(\d+\.?\d*)"
    ]

    for pattern in metric_patterns:
        metric_match = re.search(pattern, line, re.IGNORECASE)
        if metric_match:
            metric_value = float(metric_match.group(1))

            # Determine metric name based on the pattern that matched
            if "accuracy" in pattern.lower():
                metric_name = "accuracy"
            elif "test_metric" in pattern.lower():
                metric_name = "test_metric"
            elif "train_loss" in pattern.lower():
                metric_name = "train_loss"
            elif "test_loss" in pattern.lower():
                metric_name = "test_loss"
            elif "loss" in pattern.lower():
                metric_name = "loss"
            else:
                metric_name = "unknown"

            return {
                "round": round_num,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "scope": "global"
            }

    # Look for evaluation results in the format like "Results: {accuracy: 0.8, loss: 0.3}"
    eval_match = re.search(r'Results: (.+)', line)
    if eval_match:
        eval_str = eval_match.group(1)
        # Try to parse results dict - handle newlines in keys and values
        try:
            # The eval_str is a representation of a Python dict, not JSON
            # It may contain newlines in keys and values like: {'\ntest_loss': 14.449026107788086, 'test_acc\n': 0.13750000298023224, ...}

            # First, try to safely evaluate it as a Python literal
            try:
                # Use ast.literal_eval to safely parse the Python dict representation
                eval_dict = ast.literal_eval(eval_str)

                for metric_name, value in eval_dict.items():
                    if isinstance(value, (int, float)):
                        # Clean up metric name by removing newlines
                        clean_metric_name = str(metric_name).replace('\n', '').strip()
                        return {
                            "round": round_num,
                            "metric_name": clean_metric_name,
                            "metric_value": float(value),
                            "scope": "global"
                        }
            except (ValueError, SyntaxError):
                # If ast.literal_eval fails, try alternative parsing
                # Replace single quotes with double quotes to make it JSON-like and handle newlines
                cleaned_str = eval_str.strip()

                # Replace newlines in the string representation
                cleaned_str = cleaned_str.replace('\\n', '_newline')
                cleaned_str = cleaned_str.replace("'", '"')

                # Handle extra commas that might be generated
                cleaned_str = re.sub(r',\s*}', '}', cleaned_str)
                cleaned_str = re.sub(r',\s*]', ']', cleaned_str)

                eval_dict = json.loads(cleaned_str)

                for metric_name, value in eval_dict.items():
                    if isinstance(value, (int, float)):
                        # Clean up metric name by removing added "_newline" suffixes
                        clean_metric_name = metric_name.replace('_newline', '').replace('\\n', '').strip()
                        return {
                            "round": round_num,
                            "metric_name": clean_metric_name,
                            "metric_value": float(value),
                            "scope": "global"
                        }
        except json.JSONDecodeError:
            # Try alternative parsing approach - extract key-value pairs manually
            # The Results string has the format like: {'\ntest_loss': 14.449026107788086, 'test_acc\n': 0.13750000298023224, ...}
            try:
                # Pattern to match 'key': value pairs in the results, allowing for newlines in keys
                # This handles patterns like '\ntest_loss': value or 'test_acc\n': value
                # First remove the outer braces
                inner_content = eval_str.strip().strip('{}')

                # Match key-value pairs: 'key': value
                # Pattern matches: 'key': value where key may contain newlines
                kv_pairs = re.findall(r"'([^']*?)'\s*:\s*([0-9.]+)", inner_content)

                for key, value in kv_pairs:
                    # Clean the key by removing newlines from the key name
                    clean_key = key.replace('\n', '').strip()
                    if clean_key:  # Only if there's a clean key
                        return {
                            "round": round_num,
                            "metric_name": clean_key,
                            "metric_value": float(value),
                            "scope": "global"
                        }
            except:
                pass  # If alternative parsing fails, continue

            # If we can't parse the results, skip this line
            pass

    return None


def contains_relevant_info(line: str) -> bool:
    """
    Check if a line contains information relevant to convergence analysis.

    Args:
        line: A log line to check

    Returns:
        True if the line contains relevant information, False otherwise
    """
    # Keywords that indicate relevant information
    relevant_keywords = [
        'round', 'Round', 'accuracy', 'loss', 'metric', 'evaluated',
        'results', 'Results', 'test_', 'train_', 'val_', 'validation'
    ]

    # Keywords that indicate irrelevant information (to be filtered out)
    irrelevant_keywords = [
        'gossip', 'Gossip', 'beat', 'Beat', 'connect', 'Connect',
        'neighbor', 'Neighbor', 'P2P', 'p2p', 'gRPC', 'grpc',
        'socket', 'Socket', 'TLS', 'ssl', 'SSL', 'thread', 'Thread',
        'debug', 'DEBUG', 'heartbeat', 'Heartbeat', 'network', 'Network'
    ]

    line_lower = line.lower()

    # Check for irrelevant keywords first
    for keyword in irrelevant_keywords:
        if keyword.lower() in line_lower:
            return False

    # Check for relevant keywords
    for keyword in relevant_keywords:
        if keyword.lower() in line_lower:
            return True

    return False


def write_output(filtered_lines: List[str], output_file: str, output_format: str = "csv", normalized_data: List[Dict] = None):
    """
    Write the filtered and normalized output to a file.

    Args:
        filtered_lines: List of filtered lines (some commented)
        output_file: Path to save the output
        output_format: Format for output ('csv' or 'jsonl')
        normalized_data: List of normalized records for CSV output
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        if output_format == "csv" and normalized_data:
            # Write CSV header
            f.write("round,metric_name,metric_value,scope\n")
            # Write CSV data
            for record in normalized_data:
                f.write(f"{record['round']},{record['metric_name']},{record['metric_value']},{record['scope']}\n")
        else:
            # Write the filtered lines (either as JSONL or commented format)
            for line in filtered_lines:
                f.write(f"{line}\n")


def load_csv_data(csv_file: str) -> List[Dict]:
    """
    Load data from a CSV file with round, metric_name, metric_value, and scope.

    Args:
        csv_file: Path to the CSV file

    Returns:
        List of dictionaries with the data
    """
    data = []
    with open(csv_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Handle 'None' string in round column by converting to None type
            round_val = row['round']
            if round_val == 'None':
                round_num = None
            else:
                round_num = int(round_val)

            data.append({
                'round': round_num,
                'metric_name': row['metric_name'],
                'metric_value': float(row['metric_value']),
                'scope': row['scope']
            })
    return data


def plot_convergence_comparison(file1: str, file2: str, output_file: str = "convergence_comparison.png"):
    """
    Plot convergence comparison from two filtered CSV files.

    Args:
        file1: Path to first CSV file with normalized data
        file2: Path to second CSV file with normalized data
        output_file: Output file for the plot
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    # Load data from both files
    data1 = load_csv_data(file1)
    data2 = load_csv_data(file2)

    # Convert to DataFrames
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)

    # Get unique metric names from both datasets
    metrics1 = set(df1['metric_name'].unique())
    metrics2 = set(df2['metric_name'].unique())
    all_metrics = metrics1.union(metrics2)

    # Create subplots for each metric
    fig, axes = plt.subplots(len(all_metrics), 1, figsize=(12, 5*len(all_metrics)))
    if len(all_metrics) == 1:
        axes = [axes]

    for i, metric in enumerate(sorted(all_metrics)):
        ax = axes[i]

        # Filter data for this metric
        if metric in df1['metric_name'].values:
            df1_metric = df1[df1['metric_name'] == metric]
            ax.plot(df1_metric['round'], df1_metric['metric_value'],
                   label=f'Dataset 1 - {metric}', marker='o', linestyle='-', linewidth=2)

        if metric in df2['metric_name'].values:
            df2_metric = df2[df2['metric_name'] == metric]
            ax.plot(df2_metric['round'], df2_metric['metric_value'],
                   label=f'Dataset 2 - {metric}', marker='s', linestyle='-', linewidth=2)

        ax.set_xlabel('Round')
        ax.set_ylabel(metric.title())
        ax.set_title(f'{metric.title()} Convergence Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Convergence comparison plot saved to {output_file}")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Filter logs: python filter_logs.py <input_log_file> <output_file> [format:csv|jsonl]")
        print("  Compare convergence: python filter_logs.py compare <csv_file1> <csv_file2> [output_plot.png]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "compare":
        if len(sys.argv) < 4:
            print("Usage: python filter_logs.py compare <csv_file1> <csv_file2> [output_plot.png]")
            sys.exit(1)

        file1 = sys.argv[2]
        file2 = sys.argv[3]
        output_file = sys.argv[4] if len(sys.argv) > 4 else "convergence_comparison.png"

        try:
            plot_convergence_comparison(file1, file2, output_file)
        except ImportError:
            print("Error: matplotlib and pandas are required for plotting. Install with: pip install matplotlib pandas")
            sys.exit(1)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error creating plot: {e}")
            sys.exit(1)

    else:
        # Original functionality: filter and normalize logs
        if len(sys.argv) < 3:
            print("Usage: python filter_logs.py <input_log_file> <output_file> [format:csv|jsonl]")
            sys.exit(1)

        input_file = command
        output_file = sys.argv[2]
        output_format = sys.argv[3] if len(sys.argv) > 3 else "csv"

        if output_format not in ["csv", "jsonl"]:
            print(f"Invalid format: {output_format}. Use 'csv' or 'jsonl'.")
            sys.exit(1)

        try:
            filtered_lines, normalized_data = filter_and_normalize_logs(input_file, output_format)
            write_output(filtered_lines, output_file, output_format, normalized_data)
            print(f"Successfully processed {len(normalized_data)} metric records from {input_file} to {output_file}")
        except FileNotFoundError:
            print(f"Error: Input file {input_file} not found.")
            sys.exit(1)
        except Exception as e:
            print(f"Error processing file: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()