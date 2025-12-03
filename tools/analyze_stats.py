import json
import os
import glob
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import tqdm
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Analyze JSON files for specified keys.")
parser.add_argument("--json_base", type=str, help="Base directory containing JSON files.")
parser.add_argument("--output_dir", type=str, default="stat", help="Directory to save output plots.")
args = parser.parse_args()

# Configuration
json_base = args.json_base
output_dir = args.output_dir
important_keys = [
    "primaryCultureCode", 
    "year",
    "month",
    "ratingScore",
    "favoritedTimes",
    "lengthSeconds"
]

def analyze_json_files(base_path, keys):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Find all JSON files
    json_files = glob.glob(os.path.join(base_path, "*.json"))
    print(f"Found {len(json_files)} JSON files in {base_path}")

    # Data storage
    # For numeric values: store all values to calculate min/max and plot histogram
    # For string values: store counts of each unique string
    data_store = defaultdict(list)
    
    # We need to know the type of each key to decide how to process it later
    # However, we can infer it from the data collected. 
    # But to be safe, let's keep track of types we've seen.
    key_types = {}

    for file_path in tqdm.tqdm(json_files, desc="Processing files"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for key in keys:
                if key in data:
                    value = data[key]
                    
                    # Skip None values
                    if value is None:
                        continue

                    # Determine type and store
                    if isinstance(value, (int, float)):
                        data_store[key].append(value)
                        if key not in key_types:
                            key_types[key] = 'numeric'
                    elif isinstance(value, str):
                        data_store[key].append(value)
                        if key not in key_types:
                            key_types[key] = 'string'
                    else:
                        # Handle other types if necessary, or skip
                        pass
                        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Analyze and Report
    print("\n" + "="*30)
    print("Analysis Report")
    print("="*30 + "\n")

    for key in keys:
        if key not in data_store or not data_store[key]:
            print(f"Key: {key} - No data found.")
            continue

        values = data_store[key]
        k_type = key_types.get(key, 'unknown')

        print(f"Key: {key} (Type: {k_type})")
        
        if k_type == 'numeric':
            min_val = min(values)
            max_val = max(values)
            avg_val = sum(values) / len(values)
            print(f"  Min: {min_val}")
            print(f"  Max: {max_val}")
            print(f"  Avg: {avg_val:.2f}")
            print(f"  Count: {len(values)}")
            
            # Plot Histogram
            plt.figure(figsize=(10, 6))
            
            # Check for long tail distribution (X-axis)
            # Heuristic: if the right tail (max - avg) is much longer than the left tail (avg - min)
            # This indicates a strong skew towards larger values.
            is_long_tail_x = (max_val - avg_val) > 10 * (avg_val - min_val)
            
            plot_values = values
            bins = 50
            
            if is_long_tail_x:
                print("  Detected long-tail distribution (X-axis). Using log scale for x-axis.")
                # Handle 0 or negative values for log scale
                pos_values = [v for v in values if v > 0]
                if pos_values:
                    min_pos = min(pos_values)
                    # Create log-spaced bins
                    bins = np.logspace(np.log10(min_pos), np.log10(max_val), 50)
                    plt.xscale('log')
                    plot_values = pos_values
                    
                    zeros_count = len(values) - len(pos_values)
                    if zeros_count > 0:
                        print(f"  Note: {zeros_count} values <= 0 excluded from log-scale histogram.")
                else:
                    # Fallback if no positive values
                    is_long_tail_x = False
            
            # Check for uneven frequency distribution (Y-axis)
            # Heuristic: if the max count is significantly larger than the average count
            counts, _ = np.histogram(plot_values, bins=bins)
            is_long_tail_y = False
            if len(counts) > 0 and np.mean(counts) > 0:
                if max(counts) > 20 * np.mean(counts):
                    is_long_tail_y = True
                    print("  Detected uneven frequency distribution. Using log scale for y-axis.")

            plt.hist(plot_values, bins=bins, color='skyblue', edgecolor='black', log=is_long_tail_y)

            title = f"Distribution of {key}"
            if is_long_tail_x: title += " (Log X)"
            if is_long_tail_y: title += " (Log Y)"
            plt.title(title)
            plt.xlabel(key)
            plt.ylabel("Frequency")
            plt.grid(axis='y', alpha=0.75)
            
            plot_path = os.path.join(output_dir, f"{key}_histogram.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"  Histogram saved to {plot_path}")

        elif k_type == 'string':
            counts = Counter(values)
            print(f"  Unique values: {len(counts)}")
            print("  Top 5 values:")
            for val, count in counts.most_common(5):
                print(f"    {val}: {count}")
            
            # Plot Bar Chart for top 20
            top_n = 5
            most_common = counts.most_common(top_n)
            labels, sizes = zip(*most_common)
            
            plt.figure(figsize=(12, 6))
            y_pos = np.arange(len(labels))
            plt.bar(y_pos, sizes, align='center', alpha=0.7)
            plt.xticks(y_pos, labels, rotation=45, ha='right')
            plt.ylabel('Count')
            plt.title(f"Top {top_n} values for {key}")
            plt.tight_layout()
            
            plot_path = os.path.join(output_dir, f"{key}_barchart.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"  Bar chart saved to {plot_path}")
        
        print("-" * 20)

if __name__ == "__main__":
    analyze_json_files(json_base, important_keys)
