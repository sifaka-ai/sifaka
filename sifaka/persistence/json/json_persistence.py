"""
JSON persistence provider for Sifaka.

This module provides a simple JSON-based persistence provider for storing thoughts.
"""

import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from sifaka.core.interfaces import PersistenceProvider
from sifaka.core.thought import Thought


class JSONPersistence(PersistenceProvider):
    """
    JSON-based persistence provider.

    Stores thoughts as JSON files in a directory.
    """

    def __init__(self, directory: str = "./thoughts"):
        """
        Initialize the JSON persistence provider.

        Args:
            directory: Directory to store JSON files in.
        """
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        """Return the name of the persistence provider."""
        return "json"

    def save(self, thought: Thought) -> str:
        """
        Save a thought to a JSON file.

        Args:
            thought: The thought to save.

        Returns:
            A unique identifier for the saved thought.
        """
        # Generate a unique ID if not already in metadata
        thought_id = thought.metadata.get("thought_id", str(uuid.uuid4()))

        # Ensure the thought ID is in the metadata
        thought.metadata["thought_id"] = thought_id

        # Convert thought to dictionary
        data = thought.to_dict()

        # Save to file
        file_path = self.directory / f"{thought_id}.json"
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

        return thought_id

    def load(self, thought_id: str) -> Thought:
        """
        Load a thought from a JSON file.

        Args:
            thought_id: The unique identifier of the thought to load.

        Returns:
            The loaded thought.
        """
        file_path = self.directory / f"{thought_id}.json"

        if not file_path.exists():
            raise FileNotFoundError(f"Thought with ID {thought_id} not found")

        with open(file_path, "r") as f:
            data = json.load(f)

        return Thought.from_dict(data)

    def list(self, filter_criteria: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        List thought IDs in the persistence store.

        Args:
            filter_criteria: Optional criteria to filter thoughts.

        Returns:
            A list of thought IDs.
        """
        # Get all JSON files in the directory
        thought_ids = [file.stem for file in self.directory.glob("*.json") if file.is_file()]

        # If no filter criteria, return all thought IDs
        if not filter_criteria:
            return thought_ids

        # Filter thoughts based on criteria
        filtered_ids = []
        for thought_id in thought_ids:
            try:
                thought = self.load(thought_id)

                # Check if thought matches all criteria
                matches = True
                for key, value in filter_criteria.items():
                    # Handle nested keys with dot notation (e.g., "metadata.user_id")
                    keys = key.split(".")
                    current = thought.to_dict()

                    for k in keys:
                        if k not in current:
                            matches = False
                            break
                        current = current[k]

                    if current != value:
                        matches = False
                        break

                if matches:
                    filtered_ids.append(thought_id)
            except Exception:
                # Skip thoughts that can't be loaded or don't match criteria
                continue

        return filtered_ids

    def delete(self, thought_id: str) -> bool:
        """
        Delete a thought from the persistence store.

        Args:
            thought_id: The unique identifier of the thought to delete.

        Returns:
            True if the thought was deleted, False otherwise.
        """
        file_path = self.directory / f"{thought_id}.json"

        if not file_path.exists():
            return False

        os.remove(file_path)
        return True
