#!/usr/bin/env python3
"""
Dataset Explorer TUI

A terminal UI for exploring HuggingFace datasets sample by sample.
Useful for understanding what's actually in your pretraining data.

Usage:
    python explorer.py                           # Default: C4 en dataset
    python explorer.py --dataset allenai/c4 --config en
    python explorer.py --dataset PleIAs/SYNTH
    python explorer.py --dataset HuggingFaceFW/fineweb-edu --config sample-100BT
"""

import argparse
import hashlib
import json
import multiprocessing
import random
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    Footer,
    Header,
    Label,
    Static,
    Input,
    Checkbox,
)
from textual.screen import ModalScreen
from datasets import load_dataset, IterableDataset

from .config import Config

# Fix for Python 3.13 multiprocessing issue in TUI environments
# Disable tqdm's multiprocessing locks which cause issues in terminal UIs
import os
os.environ['TQDM_DISABLE'] = '0'  # Don't disable tqdm, just disable locks

# Monkey-patch tqdm to disable multiprocessing locks
try:
    from tqdm import std as tqdm_std
    # Override the lock creation to use a no-op lock
    class DummyLock:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def acquire(self, *args, **kwargs):
            pass
        def release(self, *args, **kwargs):
            pass

    tqdm_std.TqdmDefaultWriteLock.create_mp_lock = classmethod(lambda cls: setattr(cls, 'mp_lock', DummyLock()))
    tqdm_std.TqdmDefaultWriteLock.mp_lock = DummyLock()
except (ImportError, AttributeError):
    # If patching fails, continue anyway
    pass


# Default classification categories
DEFAULT_CATEGORIES = [
    "educational",
    "advertising",
    "forum",
    "news",
    "creative",
    "web_boilerplate",
    "nonsensical",
]


@dataclass
class SampleRecord:
    """A record of a classified sample."""
    dataset: str
    config: str
    sample_index: int
    sample_hash: str  # Hash for deduplication
    classification: str  # Single category
    text_preview: str  # First 500 chars for display
    full_sample: dict = field(default_factory=dict)  # Store full sample data
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "config": self.config,
            "sample_index": self.sample_index,
            "sample_hash": self.sample_hash,
            "classification": self.classification,
            "text_preview": self.text_preview,
            "full_sample": self.full_sample,
            "timestamp": self.timestamp,
        }


class ClassifyScreen(ModalScreen[Optional[str]]):
    """Modal for classifying a sample into one category."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("1", "classify_1", "1", show=False),
        Binding("2", "classify_2", "2", show=False),
        Binding("3", "classify_3", "3", show=False),
        Binding("4", "classify_4", "4", show=False),
        Binding("5", "classify_5", "5", show=False),
        Binding("6", "classify_6", "6", show=False),
        Binding("7", "classify_7", "7", show=False),
        Binding("8", "classify_8", "8", show=False),
        Binding("9", "classify_9", "9", show=False),
    ]

    def __init__(self, categories: list[str]):
        super().__init__()
        self.categories = categories

    def compose(self) -> ComposeResult:
        with Container(id="classify-dialog"):
            yield Label("Classify this sample:", id="classify-label")
            yield Label("Press number key or click to select category", id="classify-help")
            with Vertical(id="category-buttons"):
                for i, category in enumerate(self.categories, 1):
                    yield Button(
                        f"[{i}] {category}",
                        id=f"cat-{category}",
                        classes="category-btn"
                    )
            with Horizontal(id="classify-actions"):
                yield Button("Skip [Esc]", id="skip-btn")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "skip-btn":
            self.dismiss(None)
        elif event.button.id.startswith("cat-"):
            category = event.button.id[4:]
            self.dismiss(category)

    def _classify_by_index(self, index: int) -> None:
        """Classify by category index (1-based)."""
        if 1 <= index <= len(self.categories):
            self.dismiss(self.categories[index - 1])

    def action_classify_1(self) -> None:
        self._classify_by_index(1)

    def action_classify_2(self) -> None:
        self._classify_by_index(2)

    def action_classify_3(self) -> None:
        self._classify_by_index(3)

    def action_classify_4(self) -> None:
        self._classify_by_index(4)

    def action_classify_5(self) -> None:
        self._classify_by_index(5)

    def action_classify_6(self) -> None:
        self._classify_by_index(6)

    def action_classify_7(self) -> None:
        self._classify_by_index(7)

    def action_classify_8(self) -> None:
        self._classify_by_index(8)

    def action_classify_9(self) -> None:
        self._classify_by_index(9)

    def action_cancel(self) -> None:
        self.dismiss(None)


class ReviewScreen(ModalScreen[None]):
    """Modal for reviewing classified samples."""

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("n", "next_record", "Next"),
        Binding("p", "prev_record", "Previous"),
        Binding("d", "delete_record", "Delete"),
    ]

    def __init__(self, records: list[SampleRecord], text_field: str, on_delete: callable = None):
        super().__init__()
        self.records = records
        self.text_field = text_field
        self.current_index = 0
        self.on_delete_callback = on_delete

    def compose(self) -> ComposeResult:
        with Container(id="review-dialog"):
            yield Label("Classified Samples Review", id="review-title")
            yield Label("", id="review-info")
            yield Static("", id="review-content")
            with Horizontal(id="review-buttons"):
                yield Button("Previous [p]", id="prev-record-btn")
                yield Button("Next [n]", id="next-record-btn")
                yield Button("Delete [d]", variant="error", id="delete-record-btn")
                yield Button("Close [Esc]", id="close-review-btn")

    def on_mount(self) -> None:
        """Initialize the review screen."""
        self._update_display()

    def _update_display(self) -> None:
        """Update the display with current record."""
        if not self.records:
            self.query_one("#review-info", Label).update("No classified samples")
            self.query_one("#review-content", Static).update("")
            return

        record = self.records[self.current_index]
        info = f"Record {self.current_index + 1}/{len(self.records)} | Class: {record.classification} | {record.timestamp[:10]}"
        self.query_one("#review-info", Label).update(info)

        # Display full sample
        text = record.full_sample.get(self.text_field, "")
        if not text and record.full_sample:
            # If text_field not found, show all fields
            text = "\n".join(f"{k}: {v}" for k, v in record.full_sample.items())
        self.query_one("#review-content", Static).update(str(text))

    def action_next_record(self) -> None:
        """Show next record."""
        if self.records and self.current_index < len(self.records) - 1:
            self.current_index += 1
            self._update_display()

    def action_prev_record(self) -> None:
        """Show previous record."""
        if self.current_index > 0:
            self.current_index -= 1
            self._update_display()

    def action_delete_record(self) -> None:
        """Delete current record."""
        if self.records:
            deleted = self.records.pop(self.current_index)
            if self.current_index >= len(self.records) and self.current_index > 0:
                self.current_index -= 1
            self._update_display()
            # Trigger save after deletion
            if self.on_delete_callback:
                self.on_delete_callback(deleted)

    def action_close(self) -> None:
        """Close the review screen."""
        self.dismiss()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "next-record-btn":
            self.action_next_record()
        elif event.button.id == "prev-record-btn":
            self.action_prev_record()
        elif event.button.id == "delete-record-btn":
            self.action_delete_record()
        elif event.button.id == "close-review-btn":
            self.action_close()


class FieldSelectionScreen(ModalScreen[list[str]]):
    """Modal for selecting which fields to expand."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("enter", "submit", "Submit"),
    ]

    def __init__(self, fields: list[str], current_expanded: Optional[list[str]] = None):
        super().__init__()
        self.fields = fields
        self.current_expanded = current_expanded or []

    def compose(self) -> ComposeResult:
        with Container(id="field-dialog"):
            yield Label(
                "Select fields to expand to full length:",
                id="field-label"
            )
            yield Label(
                "(Use arrow keys and space to select, Enter to confirm)",
                id="field-help"
            )
            with VerticalScroll(id="field-list"):
                for field in self.fields:
                    checked = field in self.current_expanded
                    yield Checkbox(field, value=checked, id=f"field-{field}")
            with Horizontal(id="field-buttons"):
                yield Button("Save", variant="primary", id="save-fields")
                yield Button("Cancel", id="cancel-fields")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save-fields":
            self.action_submit()
        elif event.button.id == "cancel-fields":
            self.action_cancel()

    def action_submit(self) -> None:
        selected = []
        for field in self.fields:
            checkbox = self.query_one(f"#field-{field}", Checkbox)
            if checkbox.value:
                selected.append(field)
        self.dismiss(selected)

    def action_cancel(self) -> None:
        self.dismiss(self.current_expanded)


class DatasetExplorer(App):
    """TUI for exploring HuggingFace datasets."""
    
    CSS = """
    Screen {
        background: $surface;
    }
    
    #main-container {
        height: 100%;
        padding: 1;
    }
    
    #sample-view {
        height: 1fr;
        border: solid $primary;
        padding: 1;
        overflow-y: auto;
    }
    
    #stats-bar {
        height: 3;
        padding: 0 1;
        background: $boost;
    }
    
    #stats-bar Label {
        margin-right: 2;
    }
    
    #controls {
        height: 3;
        padding: 0 1;
    }
    
    #controls Button {
        margin-right: 1;
    }
    
    #classify-dialog {
        width: 50;
        height: auto;
        padding: 1 2;
        background: $surface;
        border: thick $primary;
        align: center middle;
    }

    #classify-label {
        text-style: bold;
        margin-bottom: 1;
    }

    #classify-help {
        color: $text-muted;
        margin-bottom: 1;
    }

    #category-buttons {
        height: auto;
        margin-bottom: 1;
    }

    .category-btn {
        width: 100%;
        margin-bottom: 1;
    }

    #classify-actions {
        margin-top: 1;
    }

    #current-classification {
        color: $success;
    }

    .field-name {
        color: $primary;
        text-style: bold;
    }

    #field-dialog {
        width: 60;
        height: auto;
        max-height: 80%;
        padding: 1 2;
        background: $surface;
        border: thick $primary;
    }

    #field-label {
        margin-bottom: 1;
        text-style: bold;
    }

    #field-help {
        margin-bottom: 1;
        color: $text-muted;
    }

    #field-list {
        height: auto;
        max-height: 20;
        margin-bottom: 1;
        border: solid $primary;
        padding: 1;
    }

    #field-buttons {
        margin-top: 1;
    }

    #review-dialog {
        width: 90%;
        height: 90%;
        padding: 1 2;
        background: $surface;
        border: thick $primary;
        align: center middle;
    }

    #review-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }

    #review-info {
        margin-bottom: 1;
        color: $text-muted;
    }

    #review-content {
        height: 1fr;
        border: solid $primary;
        padding: 1;
        overflow-y: auto;
        margin-bottom: 1;
    }

    #review-buttons {
        height: auto;
    }

    #review-buttons Button {
        margin-right: 1;
    }
    """
    
    BINDINGS = [
        Binding("n", "next_sample", "Next"),
        Binding("p", "prev_sample", "Previous"),
        Binding("r", "random_sample", "Random"),
        Binding("c", "classify_sample", "Classify"),
        Binding("v", "review_classified", "Review"),
        Binding("f", "configure_fields", "Fields"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        dataset_name: str = "allenai/c4",
        config: Optional[str] = "en",
        text_field: str = "text",
        categories: Optional[list[str]] = None,
        shuffle_seed: Optional[int] = None,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.config = config
        self.text_field = text_field
        self.categories = categories or DEFAULT_CATEGORIES

        # Generate session file name from dataset
        safe_name = re.sub(r'[^\w\-]', '_', dataset_name)
        if config:
            safe_name = f"{safe_name}_{config}"
        self.session_file = Path(f"{safe_name}_classifications.json")

        # Use deterministic seed for reproducibility (allows resume)
        self.shuffle_seed = shuffle_seed or hash(f"{dataset_name}_{config}") % (2**31)

        self.dataset: Optional[IterableDataset] = None
        self.current_sample: Optional[dict] = None
        self.sample_index = 0
        self.samples_viewed = 0
        self.history: list[dict] = []  # Recent samples for prev navigation
        self.history_index = -1
        self.records: list[SampleRecord] = []
        self.classified_hashes: set[str] = set()  # Track classified samples

        # Config for field expansion
        self.user_config = Config()
        self.expanded_fields: list[str] = []
        self.needs_field_config = False

        # Load existing session if available
        self._load_session()
    
    def _load_session(self) -> None:
        """Load previous session data if it exists."""
        if self.session_file.exists():
            try:
                with open(self.session_file) as f:
                    data = json.load(f)
                    self.records = [
                        SampleRecord(**r) for r in data.get("records", [])
                    ]
                    self.samples_viewed = data.get("samples_viewed", 0)
                    # Rebuild classified hashes set for deduplication
                    self.classified_hashes = {r.sample_hash for r in self.records}
            except Exception as e:
                self.notify(f"Could not load session: {e}", severity="warning")

    def _save_session(self) -> None:
        """Save session data to file."""
        # Calculate classification counts
        class_counts = {}
        for record in self.records:
            class_counts[record.classification] = class_counts.get(record.classification, 0) + 1

        data = {
            "dataset": self.dataset_name,
            "config": self.config,
            "categories": self.categories,
            "samples_viewed": self.samples_viewed,
            "classification_counts": class_counts,
            "records": [r.to_dict() for r in self.records],
            "last_saved": datetime.now().isoformat(),
        }
        with open(self.session_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="main-container"):
            with Horizontal(id="stats-bar"):
                yield Label(f"Dataset: {self.dataset_name}", id="dataset-label")
                yield Label(f"Viewed: {self.samples_viewed}", id="viewed-label")
                yield Label(f"Classified: {len(self.records)}", id="classified-label")
                yield Label("", id="current-classification")
            yield Static("Loading dataset...", id="sample-view")
            with Horizontal(id="controls"):
                yield Button("Next [n]", id="next-btn", variant="primary")
                yield Button("Prev [p]", id="prev-btn")
                yield Button("Random [r]", id="random-btn")
                yield Button("Classify [c]", id="classify-btn", variant="success")
                yield Button("Review [v]", id="review-btn")
        yield Footer()
    
    async def on_mount(self) -> None:
        """Load dataset when app starts."""
        self.title = f"Dataset Explorer - {self.dataset_name}"
        await self._load_dataset()
        await self._fetch_next_sample()

        # Check if we need to configure expanded fields
        expanded = self.user_config.get_expanded_fields(self.dataset_name, self.config)
        if expanded is None:
            self.needs_field_config = True
            # Wait for first sample to get field names, then show config in a worker
            if self.current_sample:
                self.run_worker(self._show_field_config())
        else:
            self.expanded_fields = expanded
    
    async def _load_dataset(self) -> None:
        """Load the dataset in streaming mode with deterministic shuffle."""
        try:
            # Show resume info if we have existing data
            if self.records:
                self.notify(f"Resuming: {len(self.records)} samples already classified", timeout=3)
            else:
                self.notify(f"Loading {self.dataset_name}...", timeout=2)

            if self.config:
                self.dataset = load_dataset(
                    self.dataset_name,
                    self.config,
                    split="train",
                    streaming=True,
                    trust_remote_code=False,
                )
            else:
                self.dataset = load_dataset(
                    self.dataset_name,
                    split="train",
                    streaming=True,
                    trust_remote_code=False,
                )
            # Use deterministic seed for reproducibility
            self.dataset_iter = iter(self.dataset.shuffle(seed=self.shuffle_seed))
            self.notify(f"Dataset loaded! (shuffle seed: {self.shuffle_seed})", timeout=2)
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()

            # Write error to file for debugging
            with open("/tmp/ptviewer_error.log", "w") as f:
                f.write(error_details)

            self.query_one("#sample-view", Static).update(
                f"Error loading dataset: {e}\n\n"
                f"Full traceback:\n{error_details}\n\n"
                "Make sure the dataset exists and you have internet access.\n\n"
                "(Full error log written to /tmp/ptviewer_error.log)"
            )
    
    def _compute_sample_hash(self, sample: dict) -> str:
        """Compute a hash for a sample to detect duplicates."""
        text = sample.get(self.text_field, "")
        if isinstance(text, str):
            return hashlib.md5(text[:1000].encode()).hexdigest()
        return hashlib.md5(str(sample)[:1000].encode()).hexdigest()

    async def _fetch_next_sample(self) -> None:
        """Fetch the next sample from the dataset, skipping already classified ones."""
        if self.dataset is None:
            return

        try:
            # Keep fetching until we find an unclassified sample
            max_skips = 1000  # Prevent infinite loops
            skipped = 0
            while skipped < max_skips:
                self.current_sample = next(self.dataset_iter)
                self.sample_index += 1
                self.samples_viewed += 1

                # Check if already classified
                sample_hash = self._compute_sample_hash(self.current_sample)
                if sample_hash not in self.classified_hashes:
                    break
                skipped += 1

            if skipped >= max_skips:
                self.query_one("#sample-view", Static).update(
                    "No more unclassified samples found!\n\n"
                    f"You've classified {len(self.records)} samples."
                )
                return

            # Add to history
            self.history.append(self.current_sample)
            if len(self.history) > 100:  # Keep last 100
                self.history.pop(0)
            self.history_index = len(self.history) - 1

            self._update_display()
        except StopIteration:
            # Dataset exhausted - notify user
            self.query_one("#sample-view", Static).update(
                "Dataset exhausted!\n\n"
                f"You've classified {len(self.records)} samples.\n"
                "Press 'v' to review your classifications."
            )
        except Exception as e:
            self.query_one("#sample-view", Static).update(f"Error fetching sample: {e}")
    
    async def _show_field_config(self) -> None:
        """Show field configuration dialog."""
        if not self.current_sample:
            return

        fields = list(self.current_sample.keys())
        selected = await self.push_screen_wait(
            FieldSelectionScreen(fields, self.expanded_fields)
        )
        self.expanded_fields = selected
        self.user_config.set_expanded_fields(self.dataset_name, self.config, selected)
        self.needs_field_config = False
        self._update_display()
        self.notify(f"Expanded fields saved: {', '.join(selected)}", timeout=3)

    def _update_display(self) -> None:
        """Update the display with current sample."""
        if self.current_sample is None:
            return

        # Build display text
        lines = []

        # Show all fields
        for key, value in self.current_sample.items():
            # Check if this field should be expanded
            should_expand = key in self.expanded_fields

            if isinstance(value, str):
                if should_expand:
                    # Show full value
                    lines.append(f"[bold cyan]{key}:[/]")
                    lines.append(value)
                    lines.append("")
                else:
                    # Truncate to one line
                    if len(value) > 100:
                        value = value[:100] + "..."
                    lines.append(f"[bold cyan]{key}:[/] {value}")
            else:
                lines.append(f"[bold cyan]{key}:[/] {value}")

        self.query_one("#sample-view", Static).update("\n".join(lines))

        # Update stats
        self.query_one("#viewed-label", Label).update(f"Viewed: {self.samples_viewed}")
        self.query_one("#classified-label", Label).update(f"Classified: {len(self.records)}")

        # Show classification counts summary
        class_counts = {}
        for record in self.records:
            class_counts[record.classification] = class_counts.get(record.classification, 0) + 1
        if class_counts:
            counts_str = " | ".join(f"{k}: {v}" for k, v in sorted(class_counts.items()))
            self.query_one("#current-classification", Label).update(counts_str)
        else:
            self.query_one("#current-classification", Label).update("")
    
    def action_next_sample(self) -> None:
        """Move to next sample."""
        self.run_worker(self._fetch_next_sample())
    
    def action_prev_sample(self) -> None:
        """Move to previous sample in history."""
        if self.history_index > 0:
            self.history_index -= 1
            self.current_sample = self.history[self.history_index]
            self._update_display()
        else:
            self.notify("At beginning of history", severity="warning")
    
    def action_random_sample(self) -> None:
        """Jump to a random sample (new shuffle)."""
        if self.dataset:
            # Use a new random seed for variety
            new_seed = random.randint(0, 2**31)
            self.dataset_iter = iter(self.dataset.shuffle(seed=new_seed))
            self.run_worker(self._fetch_next_sample())

    @work
    async def action_classify_sample(self) -> None:
        """Open classification dialog for current sample."""
        if self.current_sample is None:
            return

        classification = await self.push_screen_wait(ClassifyScreen(self.categories))

        if classification is None:
            # User skipped - just move to next sample
            self.run_worker(self._fetch_next_sample())
            return

        # Create record
        text = self.current_sample.get(self.text_field, "")
        preview = text[:500] if isinstance(text, str) else str(text)[:500]
        sample_hash = self._compute_sample_hash(self.current_sample)

        record = SampleRecord(
            dataset=self.dataset_name,
            config=self.config or "",
            sample_index=self.sample_index,
            sample_hash=sample_hash,
            classification=classification,
            text_preview=preview,
            full_sample=self.current_sample.copy(),
        )
        self.records.append(record)
        self.classified_hashes.add(sample_hash)

        # Auto-save after each classification
        self._save_session()

        self._update_display()
        self.notify(f"Classified as: {classification} (auto-saved)", timeout=1)

        # Automatically move to next sample
        self.run_worker(self._fetch_next_sample())

    @work
    async def action_configure_fields(self) -> None:
        """Open field configuration dialog."""
        await self._show_field_config()

    def _on_record_deleted(self, deleted_record: SampleRecord) -> None:
        """Handle record deletion from review screen."""
        # Remove from classified hashes
        if deleted_record.sample_hash in self.classified_hashes:
            self.classified_hashes.remove(deleted_record.sample_hash)
        # Auto-save after deletion
        self._save_session()

    @work
    async def action_review_classified(self) -> None:
        """Open review screen for classified samples."""
        if not self.records:
            self.notify("No classified samples to review", severity="warning", timeout=2)
            return

        await self.push_screen_wait(
            ReviewScreen(self.records, self.text_field, on_delete=self._on_record_deleted)
        )
        self._update_display()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_actions = {
            "next-btn": self.action_next_sample,
            "prev-btn": self.action_prev_sample,
            "random-btn": self.action_random_sample,
            "classify-btn": self.action_classify_sample,
            "review-btn": self.action_review_classified,
        }
        action = button_actions.get(event.button.id)
        if action:
            action()


def main():
    parser = argparse.ArgumentParser(
        description="Classify HuggingFace dataset samples for training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
    # Classify C4 English samples
    ptviewer --dataset allenai/c4 --config en

    # Classify SYNTH dataset
    ptviewer --dataset PleIAs/SYNTH --text-field synthetic_answer

    # Classify fineweb-edu sample
    ptviewer --dataset HuggingFaceFW/fineweb-edu --config sample-100BT

    # Use custom categories
    ptviewer --dataset allenai/c4 --categories spam,ham,uncertain

Default categories: {', '.join(DEFAULT_CATEGORIES)}

Session files are auto-named based on dataset (e.g., allenai_c4_en_classifications.json)
        """
    )
    parser.add_argument(
        "--dataset", "-d",
        default="allenai/c4",
        help="HuggingFace dataset name (default: allenai/c4)"
    )
    parser.add_argument(
        "--config", "-c",
        default="en",
        help="Dataset config/subset (default: en, use 'none' to skip)"
    )
    parser.add_argument(
        "--text-field", "-t",
        default="text",
        help="Field containing main text (default: text)"
    )
    parser.add_argument(
        "--categories",
        default=None,
        help=f"Comma-separated classification categories (default: {','.join(DEFAULT_CATEGORIES)})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Shuffle seed for reproducibility (default: auto-generated from dataset name)"
    )

    args = parser.parse_args()

    config = None if args.config == "none" else args.config
    categories = args.categories.split(",") if args.categories else None

    app = DatasetExplorer(
        dataset_name=args.dataset,
        config=config,
        text_field=args.text_field,
        categories=categories,
        shuffle_seed=args.seed,
    )
    app.run()


if __name__ == "__main__":
    main()