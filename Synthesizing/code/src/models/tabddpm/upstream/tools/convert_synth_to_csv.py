#!/usr/bin/env python3
"""
Convert generated synthetic data from npy files to CSV format.
"""
import os
import sys
import numpy as np
import pandas as pd
import argparse

def convert_to_csv(parent_dir, output_path=None):
    """
    Convert generated synthetic data to CSV.
    
    Args:
        parent_dir: Directory containing X_num_train.npy, X_cat_train.npy, y_train.npy
        output_path: Output CSV file path (default: parent_dir/synth_train.csv)
    """
    parent_dir = os.path.abspath(parent_dir)
    
    # Load npy files
    x_num_path = os.path.join(parent_dir, 'X_num_train.npy')
    x_cat_path = os.path.join(parent_dir, 'X_cat_train.npy')
    y_path = os.path.join(parent_dir, 'y_train.npy')
    
    data_parts = []
    column_names = []
    
    # Load numerical features
    if os.path.exists(x_num_path):
        X_num = np.load(x_num_path, allow_pickle=True)
        print(f"Loaded X_num: shape {X_num.shape}")
        data_parts.append(X_num)
        # Create column names for numerical features
        for i in range(X_num.shape[1]):
            column_names.append(f'num_{i}')
    
    # Load categorical features
    if os.path.exists(x_cat_path):
        X_cat = np.load(x_cat_path, allow_pickle=True)
        print(f"Loaded X_cat: shape {X_cat.shape}")
        data_parts.append(X_cat)
        # Create column names for categorical features
        for i in range(X_cat.shape[1]):
            column_names.append(f'cat_{i}')
    
    # Load target
    if os.path.exists(y_path):
        y = np.load(y_path, allow_pickle=True)
        print(f"Loaded y: shape {y.shape}")
        # Reshape if needed
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        data_parts.append(y)
        column_names.append('y')
    
    if not data_parts:
        raise ValueError(f"No data files found in {parent_dir}")
    
    # Concatenate all parts
    data = np.hstack(data_parts)
    print(f"Combined data shape: {data.shape}")
    print(f"Number of columns: {len(column_names)}")
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=column_names)
    
    # Determine output path
    if output_path is None:
        output_path = os.path.join(parent_dir, 'synth_train.csv')
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"[OK] Saved synthetic data to: {output_path}")
    print(f"[OK] Total samples: {len(df)}, Total columns: {len(df.columns)}")
    
    # Print summary statistics
    print("\n=== Data Summary ===")
    print(df.describe())
    
    return output_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert synthetic npy files to CSV')
    parser.add_argument('parent_dir', type=str, help='Directory containing generated npy files')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output CSV file path (default: parent_dir/synth_train.csv)')
    
    args = parser.parse_args()
    convert_to_csv(args.parent_dir, args.output)
