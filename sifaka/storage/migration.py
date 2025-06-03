"""Migration utilities for Sifaka storage system.

This module provides utilities to migrate from the legacy storage system
to the new PydanticAI-native persistence system.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from sifaka.core.thought import SifakaThought
from sifaka.storage.base import SifakaBasePersistence
from sifaka.utils import get_logger

logger = get_logger(__name__)


class StorageMigrator:
    """Utility class for migrating between storage systems.

    This class helps migrate thoughts from legacy storage formats
    to the new PydanticAI-native persistence system.
    """

    def __init__(self, target_persistence: SifakaBasePersistence):
        """Initialize the migrator.

        Args:
            target_persistence: The target persistence backend to migrate to
        """
        self.target = target_persistence
        logger.debug(
            f"Initialized StorageMigrator with target: {type(target_persistence).__name__}"
        )

    async def migrate_from_legacy_file(self, file_path: str) -> int:
        """Migrate thoughts from a legacy JSON file.

        Args:
            file_path: Path to the legacy storage file

        Returns:
            Number of thoughts successfully migrated
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.error(f"Legacy file not found: {file_path}")
                return 0

            with open(file_path, "r", encoding="utf-8") as f:
                legacy_data = json.load(f)

            migrated_count = 0

            # Handle different legacy formats
            if isinstance(legacy_data, dict):
                # Key-value format from legacy storage
                for key, value in legacy_data.items():
                    if await self._migrate_single_thought(key, value):
                        migrated_count += 1
            elif isinstance(legacy_data, list):
                # List format
                for i, item in enumerate(legacy_data):
                    if await self._migrate_single_thought(f"item_{i}", item):
                        migrated_count += 1

            logger.info(f"Migrated {migrated_count} thoughts from {file_path}")
            return migrated_count

        except Exception as e:
            logger.error(f"Failed to migrate from legacy file {file_path}: {e}")
            return 0

    async def _migrate_single_thought(self, key: str, data: Any) -> bool:
        """Migrate a single thought from legacy format.

        Args:
            key: The storage key
            data: The legacy thought data

        Returns:
            True if migration was successful, False otherwise
        """
        try:
            # Try to convert legacy thought to SifakaThought
            if isinstance(data, dict):
                # Check if it looks like a legacy Thought object
                if "prompt" in data and "id" in data:
                    thought = await self._convert_legacy_thought(data)
                    if thought:
                        await self.target.store_thought(thought)
                        logger.debug(f"Migrated thought: {thought.id}")
                        return True
                else:
                    logger.warning(f"Skipping non-thought data for key: {key}")

            return False

        except Exception as e:
            logger.warning(f"Failed to migrate thought for key {key}: {e}")
            return False

    async def _convert_legacy_thought(self, legacy_data: Dict[str, Any]) -> Optional[SifakaThought]:
        """Convert legacy thought data to SifakaThought.

        Args:
            legacy_data: Legacy thought data dictionary

        Returns:
            Converted SifakaThought or None if conversion failed
        """
        try:
            # Extract basic fields
            thought_id = legacy_data.get("id", "")
            prompt = legacy_data.get("prompt", "")

            if not prompt:
                logger.warning("Skipping thought with empty prompt")
                return None

            # Create new SifakaThought
            thought = SifakaThought(
                id=thought_id if thought_id else None,  # Let it generate new ID if empty
                prompt=prompt,
                max_iterations=legacy_data.get("max_iterations", 3),
            )

            # Migrate text content
            if "text" in legacy_data:
                thought.current_text = legacy_data["text"]
                thought.final_text = legacy_data["text"]  # Assume it's final in legacy

            # Migrate metadata
            if "model_name" in legacy_data:
                # Add as a generation record
                from datetime import datetime
                from sifaka.core.thought import Generation

                generation = Generation(
                    iteration=0,
                    text=thought.current_text or "",
                    model=legacy_data["model_name"],
                    timestamp=datetime.now(),
                )
                thought.generations.append(generation)

            # Migrate validation results
            if "validation_results" in legacy_data:
                validation_results = legacy_data["validation_results"]
                if isinstance(validation_results, dict):
                    for validator_name, result in validation_results.items():
                        if isinstance(result, dict):
                            thought.add_validation(
                                validator=validator_name,
                                passed=result.get("passed", False),
                                details=result,
                            )

            # Migrate critic feedback
            if "critic_feedback" in legacy_data:
                critic_feedback = legacy_data["critic_feedback"]
                if isinstance(critic_feedback, list):
                    for feedback in critic_feedback:
                        if isinstance(feedback, dict):
                            thought.add_critique(
                                critic=feedback.get("critic", "unknown"),
                                feedback=feedback.get("feedback", ""),
                                suggestions=feedback.get("suggestions", []),
                            )

            # Migrate timestamps
            if "timestamp" in legacy_data:
                try:
                    from datetime import datetime

                    if isinstance(legacy_data["timestamp"], str):
                        thought.created_at = datetime.fromisoformat(legacy_data["timestamp"])
                    elif hasattr(legacy_data["timestamp"], "isoformat"):
                        thought.created_at = legacy_data["timestamp"]
                except Exception:
                    pass  # Keep default timestamp

            # Migrate conversation data
            if "chain_id" in legacy_data:
                thought.conversation_id = legacy_data["chain_id"]

            if "parent_id" in legacy_data:
                thought.parent_thought_id = legacy_data["parent_id"]

            logger.debug(f"Converted legacy thought: {thought.id}")
            return thought

        except Exception as e:
            logger.error(f"Failed to convert legacy thought: {e}")
            return None

    async def migrate_from_legacy_storage(self, legacy_storage: Any) -> int:
        """Migrate thoughts from a legacy storage instance.

        Args:
            legacy_storage: Legacy storage instance with get/search methods

        Returns:
            Number of thoughts successfully migrated
        """
        try:
            migrated_count = 0

            # Try to get all data from legacy storage
            if hasattr(legacy_storage, "search"):
                # Use search to get all items
                all_items = await legacy_storage.search("", limit=10000)

                for item in all_items:
                    if await self._migrate_single_thought(f"search_item", item):
                        migrated_count += 1

            elif hasattr(legacy_storage, "data"):
                # Direct access to data dictionary (MemoryStorage)
                for key, value in legacy_storage.data.items():
                    if await self._migrate_single_thought(key, value):
                        migrated_count += 1

            else:
                logger.warning("Legacy storage format not recognized for migration")

            logger.info(f"Migrated {migrated_count} thoughts from legacy storage")
            return migrated_count

        except Exception as e:
            logger.error(f"Failed to migrate from legacy storage: {e}")
            return 0

    async def export_to_file(self, file_path: str, conversation_id: Optional[str] = None) -> int:
        """Export thoughts from target persistence to a file.

        Args:
            file_path: Path to export file
            conversation_id: Optional conversation ID to filter by

        Returns:
            Number of thoughts exported
        """
        try:
            # Get thoughts from target persistence
            thoughts = await self.target.list_thoughts(conversation_id=conversation_id)

            # Convert to exportable format
            export_data = []
            for thought in thoughts:
                export_data.append(thought.model_dump())

            # Write to file
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"Exported {len(export_data)} thoughts to {file_path}")
            return len(export_data)

        except Exception as e:
            logger.error(f"Failed to export thoughts to {file_path}: {e}")
            return 0

    async def validate_migration(self, source_count: int) -> Dict[str, Any]:
        """Validate that migration was successful.

        Args:
            source_count: Expected number of thoughts from source

        Returns:
            Validation report dictionary
        """
        try:
            # Get all thoughts from target
            target_thoughts = await self.target.list_thoughts()
            target_count = len(target_thoughts)

            # Basic validation
            success_rate = (target_count / source_count * 100) if source_count > 0 else 0

            # Detailed analysis
            validation_report = {
                "source_count": source_count,
                "target_count": target_count,
                "success_rate": success_rate,
                "migration_successful": success_rate >= 90,  # 90% threshold
                "thoughts_with_text": len([t for t in target_thoughts if t.current_text]),
                "thoughts_with_validations": len([t for t in target_thoughts if t.validations]),
                "thoughts_with_critiques": len([t for t in target_thoughts if t.critiques]),
                "unique_conversations": len(
                    set(t.conversation_id for t in target_thoughts if t.conversation_id)
                ),
            }

            logger.info(f"Migration validation: {validation_report}")
            return validation_report

        except Exception as e:
            logger.error(f"Failed to validate migration: {e}")
            return {"error": str(e), "migration_successful": False}


async def migrate_legacy_file_to_persistence(
    legacy_file_path: str, target_persistence: SifakaBasePersistence
) -> Dict[str, Any]:
    """Convenience function to migrate a legacy file to new persistence.

    Args:
        legacy_file_path: Path to legacy storage file
        target_persistence: Target persistence backend

    Returns:
        Migration report dictionary
    """
    migrator = StorageMigrator(target_persistence)

    # Perform migration
    migrated_count = await migrator.migrate_from_legacy_file(legacy_file_path)

    # Validate migration
    validation_report = await migrator.validate_migration(migrated_count)

    return {
        "migrated_count": migrated_count,
        "validation": validation_report,
        "migration_successful": validation_report.get("migration_successful", False),
    }
