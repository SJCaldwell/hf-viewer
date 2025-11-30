"""Configuration management for ptviewer."""

import json
from pathlib import Path
from typing import Optional


class Config:
    """Manages user configuration for field display preferences."""

    def __init__(self, config_dir: Optional[Path] = None):
        if config_dir is None:
            config_dir = Path.home() / ".config" / "ptviewer"
        self.config_dir = config_dir
        self.config_file = config_dir / "config.json"
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self.data = self._load()

    def _load(self) -> dict:
        """Load configuration from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file) as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def save(self) -> None:
        """Save configuration to file."""
        with open(self.config_file, "w") as f:
            json.dump(self.data, indent=2, fp=f)

    def get_expanded_fields(self, dataset: str, config: Optional[str]) -> Optional[list[str]]:
        """Get expanded fields for a specific dataset/config combination."""
        key = self._dataset_key(dataset, config)
        return self.data.get("datasets", {}).get(key, {}).get("expanded_fields")

    def set_expanded_fields(self, dataset: str, config: Optional[str], fields: list[str]) -> None:
        """Set expanded fields for a specific dataset/config combination."""
        if "datasets" not in self.data:
            self.data["datasets"] = {}

        key = self._dataset_key(dataset, config)
        if key not in self.data["datasets"]:
            self.data["datasets"][key] = {}

        self.data["datasets"][key]["expanded_fields"] = fields
        self.save()

    def has_config_for(self, dataset: str, config: Optional[str]) -> bool:
        """Check if configuration exists for a dataset/config combination."""
        key = self._dataset_key(dataset, config)
        return key in self.data.get("datasets", {})

    @staticmethod
    def _dataset_key(dataset: str, config: Optional[str]) -> str:
        """Generate a unique key for dataset/config combination."""
        if config:
            return f"{dataset}::{config}"
        return dataset
