# Dataset Classifier TUI

A terminal UI for classifying HuggingFace dataset samples to create training data for text classifiers. Built for efficiently labeling hundreds of samples with automatic saving and resume support.

## Installation

```bash
uv sync
```

## Quick Start

```bash
# Classify C4 English samples (default)
uv run ptviewer

# Classify specific datasets
uv run ptviewer --dataset allenai/c4 --config en
uv run ptviewer --dataset HuggingFaceFW/fineweb-edu --config sample-100BT
uv run ptviewer --dataset PleIAs/SYNTH --config none --text-field synthetic_answer
```

## Workflow

1. **Start classifying** - Run the command for your dataset
2. **Press `c`** to classify each sample into one of 7 categories
3. **Press number keys `1-7`** or click to select a category
4. **Auto-saves** after each classification - you can quit anytime
5. **Resume anytime** - just run the same command again

The app automatically:
- Saves after every classification (no data loss if you quit/crash)
- Skips samples you've already classified
- Names session files based on dataset (e.g., `allenai_c4_en_classifications.json`)

## Default Categories

1. `educational` - Educational/instructional content
2. `advertising` - Marketing, ads, promotional content
3. `forum` - Discussion forums, Q&A, comments
4. `news` - News articles, journalism
5. `creative` - Fiction, poetry, creative writing
6. `web_boilerplate` - Navigation, menus, cookie notices, legal text
7. `nonsensical` - Garbled text, encoding errors, spam

### Custom Categories

```bash
uv run ptviewer --dataset mydata --categories spam,ham,uncertain
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `c` | Classify current sample |
| `1-7` | Quick-select category (in classify dialog) |
| `n` | Next sample |
| `p` | Previous sample (from history) |
| `r` | Random shuffle (new random samples) |
| `v` | Review classified samples |
| `f` | Configure expanded fields |
| `Esc` | Skip sample / Cancel |
| `q` | Quit |

## Session Files

Session files are automatically created based on your dataset name:

| Dataset | Session File |
|---------|--------------|
| `allenai/c4` (config: `en`) | `allenai_c4_en_classifications.json` |
| `HuggingFaceFW/fineweb-edu` | `HuggingFaceFW_fineweb-edu_sample-100BT_classifications.json` |

### Session File Format

```json
{
  "dataset": "allenai/c4",
  "config": "en",
  "categories": ["educational", "advertising", "forum", "news", "creative", "web_boilerplate", "nonsensical"],
  "samples_viewed": 150,
  "classification_counts": {
    "educational": 45,
    "news": 30,
    "forum": 25,
    "web_boilerplate": 20,
    "advertising": 15,
    "creative": 10,
    "nonsensical": 5
  },
  "records": [
    {
      "dataset": "allenai/c4",
      "config": "en",
      "sample_index": 42,
      "sample_hash": "abc123...",
      "classification": "educational",
      "text_preview": "First 500 chars...",
      "full_sample": { "text": "Full text...", "url": "..." },
      "timestamp": "2024-01-15T10:30:00"
    }
  ]
}
```

## Analyzing Results

```python
import json
from collections import Counter

with open("allenai_c4_en_classifications.json") as f:
    data = json.load(f)

# See distribution
print("Classification counts:", data["classification_counts"])

# Get samples by category
educational = [r for r in data["records"] if r["classification"] == "educational"]
print(f"Found {len(educational)} educational samples")

# Export for training
import pandas as pd
df = pd.DataFrame([
    {"text": r["full_sample"]["text"], "label": r["classification"]}
    for r in data["records"]
])
df.to_csv("training_data.csv", index=False)
```

## Field Expansion

By default, most fields are truncated to 100 characters. Press `f` to configure which fields show full content.

Your preferences are saved per dataset in `~/.config/ptviewer/config.json`.

## CLI Options

```
--dataset, -d     HuggingFace dataset name (default: allenai/c4)
--config, -c      Dataset config/subset (default: en, use 'none' to skip)
--text-field, -t  Field containing main text (default: text)
--categories      Comma-separated categories (default: educational,advertising,forum,news,creative,web_boilerplate,nonsensical)
--seed            Shuffle seed for reproducibility
```

## Dataset-Specific Notes

### C4
- Config: `en`, `en.noblocklist`, `en.noclean`, `realnewslike`
- Text field: `text`

### fineweb-edu
- Configs: `sample-10BT`, `sample-100BT`, `sample-350BT`, `default`
- Text field: `text`

### PleIAs/SYNTH
- Use `--config none`
- Text field: `synthetic_answer` or `synthetic_reasoning`

## Why This Exists

For training text classifiers, you need labeled data. This tool lets you efficiently label 200-500 samples per dataset by:

1. **Streaming random samples** - No need to download entire datasets
2. **Auto-saving progress** - Never lose work
3. **Resume support** - Label across multiple sessions
4. **Simple keyboard workflow** - Press `c`, then `1-7`, repeat

The labeled data can then be used to train a classifier to automatically categorize the rest of your dataset.
