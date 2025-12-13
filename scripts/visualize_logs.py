
import re
import os
import json
import argparse
import glob
import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Determine project root relative to this script
# Script is in <root>/scripts/visualize_logs.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# Default paths
DEFAULT_LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results")

def get_latest_log_file(logs_dir):
    # Find all run-*.log files
    search_path = os.path.join(logs_dir, "run-*.log")
    log_files = glob.glob(search_path)
    if not log_files:
        return None
    # Sort by modification time
    return max(log_files, key=os.path.getmtime)

def get_all_log_parts(base_log_path):
    """
    Finds all rotated parts of a log file (e.g., run.log.2, run.log.1, run.log)
    and returns them in chronological order (oldest -> newest).
    """
    if not os.path.exists(base_log_path):
        return []
    
    # Check for rotated files: base_log_path.<N>
    # Standard rotation: .2 is older than .1, and .1 is older than base
    
    # Find all files starting with base_log_path + "."
    directory = os.path.dirname(base_log_path)
    base_name = os.path.basename(base_log_path)
    
    rotated_files = []
    # Using glob to find candidates
    for f in glob.glob(f"{base_log_path}.*"):
        # Check if suffix is integer
        suffix = f.split('.')[-1]
        if suffix.isdigit():
            rotated_files.append((int(suffix), f))
            
    # Sort by index descending (highest index = oldest file)
    rotated_files.sort(key=lambda x: x[0], reverse=True)
    
    ordered_files = [f[1] for f in rotated_files]
    ordered_files.append(base_log_path)
    
    return ordered_files

def parse_log_files(log_paths):
    metrics = {}
    last_finished_round = {} # node -> last round num finished
    all_keys = set()
    
    for log_path in log_paths:
        print(f"Reading log file part: {log_path}")
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # Track Round progress
                # Example: [ ... | node_1 | ... ] 🎉 Round 40 of 50 finished.
                if "Round" in line and "finished" in line:
                    round_match = re.search(r'\|\s*(node(?:_\d+)?)\s*\|.*Round (\d+)', line)
                    if round_match:
                        node_name = round_match.group(1)
                        round_num = int(round_match.group(2))
                        last_finished_round[node_name] = round_num

                if "Evaluated. Results:" in line:
                    try:
                        # Extract Node
                        node_match = re.search(r'\|\s*(node(?:_\d+)?)\s*\|', line)
                        if not node_match: continue
                        node_name = node_match.group(1)
                        
                        # Determine Round Number
                        # If sequence is Train -> Eval -> Finished(X), then at Eval, we are in Round X.
                        # We haven't seen Finished(X) yet, so last_finished is X-1.
                        # So current_round = last_finished + 1.
                        current_round = last_finished_round.get(node_name, 0) + 1
                        
                        # Extract JSON
                        json_part = line.split("Results:", 1)[1].strip()
                        json_part = json_part.replace("'", '"')
                        
                        # Try to clean up json string if needed
                        end_idx = json_part.rfind('}')
                        if end_idx != -1:
                            json_part = json_part[:end_idx+1]
                        
                        data = json.loads(json_part)
                        
                        if node_name not in metrics:
                            metrics[node_name] = []
                        
                        # Collect all keys found in the data
                        current_keys = set(data.keys())
                        all_keys.update(current_keys)
                        
                        # Store tuple (round, data)
                        metrics[node_name].append((current_round, data))
                        
                    except Exception as e:
                        print(f"Skipping line due to error: {e}")
                        continue

    print(f"DEBUG: Found metrics keys: {all_keys}")
    
    return metrics, sorted(list(all_keys))

def plot_metrics_web(metrics, keys, output_path):
    if not keys:
        print("No metrics found to plot.")
        return

    # Create subplots
    num_metrics = len(keys)
    
    fig = make_subplots(
        rows=1, cols=num_metrics,
        subplot_titles=[f"<b>{k.replace('_', ' ').title()}</b>" for k in keys],
        horizontal_spacing=0.08
    )
    
    nodes = sorted(metrics.keys())
    
    for i, m_name in enumerate(keys):
        col = i + 1
        for node in nodes:
            data_points = metrics.get(node, [])
            if not data_points: continue
            
            # Extract x (Round) and y (Value) from (round, data) tuples
            values = []
            rounds = []
            
            # Sort by round
            data_points.sort(key=lambda x: x[0])
            
            for r, d in data_points:
                if m_name in d:
                    values.append(d[m_name])
                    rounds.append(r)
            
            if not values: continue

            fig.add_trace(
                go.Scatter(
                    x=rounds, 
                    y=values, 
                    mode='lines+markers', 
                    name=node,
                    line=dict(width=2),
                    marker=dict(size=6),
                    legendgroup=node, 
                    showlegend=(i==0),
                    hovertemplate=f"<b>{node}</b><br>Round: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>"
                ),
                row=1, col=col
            )
            
    # Polish the layout
    fig.update_layout(
        title={
            'text': f"<b>Training Metrics Analysis</b><br><span style='font-size: 12px; color: gray;'>File: {os.path.basename(output_path)}</span>",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        height=600,
        hovermode="closest",
        template="plotly_white",
        font=dict(
            family="Inter, Arial, sans-serif",
            size=12,
            color="#2c3e50"
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            title=dict(text="<b>Nodes</b>")
        ),
        margin=dict(l=40, r=150, t=100, b=50) # Extra right margin for legend
    )
    
    # Add grid lines and spike lines for better readability
    fig.update_xaxes(
        title_text="<b>Round</b>", 
        showgrid=True, 
        gridwidth=1, 
        gridcolor='#f0f0f0',
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        showline=True, 
        linewidth=1, 
        linecolor='black',
        mirror=True
    )
    fig.update_yaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='#f0f0f0',
        showline=True, 
        linewidth=1, 
        linecolor='black',
        mirror=True
    )
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig.write_html(output_path)
    print(f"Interactive plot saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize P2PFL Experiment Results (Web)")
    parser.add_argument("log_file", nargs="?", help="Path to the log file. If not provided, uses the latest log in default logs dir.")
    parser.add_argument("--logs-dir", default=DEFAULT_LOGS_DIR, help=f"Directory to search for logs (default: {DEFAULT_LOGS_DIR})")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help=f"Directory to save output (default: {DEFAULT_OUTPUT_DIR})")
    
    args = parser.parse_args()
    
    # Determine log file
    log_file = args.log_file
    if not log_file:
        if os.path.exists(args.logs_dir):
            print(f"Searching for latest log in {args.logs_dir}...")
            log_file = get_latest_log_file(args.logs_dir)
        else:
            print(f"Error: Logs directory '{args.logs_dir}' does not exist.")
            exit(1)
    
    if not log_file or not os.path.exists(log_file):
        print(f"Error: Log file not found: {log_file}")
        exit(1)
        
    print(f"Processing experiment logs centered on: {log_file}")
    
    # Get all parts
    all_log_parts = get_all_log_parts(log_file)
    print(f"Found {len(all_log_parts)} log file parts: {all_log_parts}")
    
    # Generate output filename
    base_name = os.path.basename(log_file)
    name_without_ext = os.path.splitext(base_name)[0]
    output_html = os.path.join(args.output_dir, f"{name_without_ext}_metrics.html")
    
    data, keys = parse_log_files(all_log_parts)
    plot_metrics_web(data, keys, output_html)
