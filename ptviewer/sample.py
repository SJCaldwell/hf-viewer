#!/usr/bin/env python3
"""
Quick dataset sampler - dump random samples to stdout.

Usage:
    python sample.py --dataset allenai/c4 --config en --count 5
    python sample.py --dataset PleIAs/SYNTH --count 10 --json
"""

import argparse
import json
import random
import sys

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Dump random samples from a HuggingFace dataset")
    parser.add_argument("--dataset", "-d", default="allenai/c4", help="Dataset name")
    parser.add_argument("--config", "-c", default="en", help="Config (use 'none' to skip)")
    parser.add_argument("--count", "-n", type=int, default=5, help="Number of samples")
    parser.add_argument("--text-field", "-t", default="text", help="Main text field")
    parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--max-chars", type=int, default=2000, help="Max chars per sample (0 for unlimited)")
    
    args = parser.parse_args()
    
    config = None if args.config == "none" else args.config
    seed = args.seed if args.seed is not None else random.randint(0, 10000)
    
    print(f"Loading {args.dataset}...", file=sys.stderr)
    
    if config:
        dataset = load_dataset(
            args.dataset, config, split="train", 
            streaming=True
        )
    else:
        dataset = load_dataset(
            args.dataset, split="train",
            streaming=True
        )
    
    dataset = dataset.shuffle(seed=seed)
    
    samples = []
    for i, sample in enumerate(dataset):
        if i >= args.count:
            break
        samples.append(sample)
    
    if args.json:
        print(json.dumps(samples, indent=2, default=str))
    else:
        for i, sample in enumerate(samples):
            print(f"\n{'='*60}")
            print(f"SAMPLE {i+1}")
            print('='*60)
            
            for key, value in sample.items():
                if key == args.text_field:
                    continue
                if isinstance(value, str) and len(value) > 100:
                    value = value[:100] + "..."
                print(f"{key}: {value}")
            
            print(f"\n{args.text_field}:")
            print("-"*40)
            text = sample.get(args.text_field, "")
            if args.max_chars and len(text) > args.max_chars:
                text = text[:args.max_chars] + f"\n... [truncated, {len(sample.get(args.text_field, ''))} total chars]"
            print(text)


if __name__ == "__main__":
    main()