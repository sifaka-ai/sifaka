"""SifakaEngine: Simplified engine with built-in features.

This module implements the core engine with simple built-in features instead
of complex middleware system.
"""

import uuid
import time
import hashlib
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from pydantic_graph import Graph
from pydantic_graph.persistence import BaseStatePersistence
from pydantic_graph.persistence.in_mem import FullStatePersistence

from sifaka.core.thought import SifakaThought
from sifaka.graph.dependencies import SifakaDependencies
from sifaka.utils.errors import GraphExecutionError
from sifaka.utils.logging import get_logger
from sifaka.utils.validation import validate_max_iterations, validate_prompt

logger = get_logger(__name__)


class SifakaEngine:
    """Main orchestration engine for Sifaka with built-in features."""

    def __init__(
        self,
        config: Optional["SifakaConfig"] = None,
        dependencies: Optional[SifakaDependencies] = None,
        persistence: Optional[BaseStatePersistence] = None,
    ):
        """Initialize the Sifaka engine."""
        # Import here to avoid circular imports
        from sifaka.utils.config import SifakaConfig

        if config is not None:
            # Create dependencies from config
            self.config = config
            self.deps = self._create_dependencies_from_config(config)
        else:
            # Use provided dependencies or create default
            self.config = None
            self.deps = dependencies or SifakaDependencies.create_default()

        # Initialize built-in features
        self._cache: Dict[str, Any] = {} if (config and config.enable_caching) else {}
        self._cache_timestamps: Dict[str, datetime] = {}
        self._timing_data: List[Dict[str, Any]] = []

        # Use PydanticAI's FullStatePersistence by default
        if persistence is None:
            self.persistence = FullStatePersistence()
        else:
            self.persistence = persistence

        # Create the PydanticAI graph with lazy node imports
        from sifaka.graph.nodes import CritiqueNode, GenerateNode, ValidateNode

        self.graph = Graph(
            nodes=[GenerateNode, ValidateNode, CritiqueNode],
            state_type=SifakaThought,
            run_end_type=SifakaThought,
            name="SifakaWorkflow",
        )

        logger.info(
            "SifakaEngine initialized",
            extra={
                "config_provided": config is not None,
                "dependencies_type": type(self.deps).__name__,
                "persistence_type": type(self.persistence).__name__,
                "logging_enabled": config.enable_logging if config else False,
                "timing_enabled": config.enable_timing if config else False,
                "caching_enabled": config.enable_caching if config else False,
            },
        )

    def _create_dependencies_from_config(self, config: "SifakaConfig") -> SifakaDependencies:
        """Create SifakaDependencies from SifakaConfig."""
        from sifaka.validators import (
            min_length_validator,
            max_length_validator,
            sentiment_validator,
        )

        # Build validators from config
        validators = []
        if config.min_length is not None:
            validators.append(min_length_validator(config.min_length))
        if config.max_length is not None:
            validators.append(max_length_validator(config.max_length))
        if config.required_sentiment is not None:
            validators.append(sentiment_validator(required_sentiments=[config.required_sentiment]))

        # Build critics dict from config - map critic names to default model names
        default_critic_models = {
            "reflexion": "openai:gpt-4o-mini",
            "constitutional": "anthropic:claude-3-5-haiku-20241022",
            "self_refine": "gemini-1.5-flash",
            "n_critics": "groq:llama-3.1-8b-instant",
            "self_consistency": "openai:gpt-3.5-turbo",
            "prompt": "anthropic:claude-3-haiku-20240307",
            "meta_rewarding": "gemini-1.5-flash",
            "self_rag": "groq:mixtral-8x7b-32768",
        }

        critics = {}
        for name in config.critics:
            # Use default model for each critic type, fallback to main model
            critics[name] = default_critic_models.get(name, config.model)

        return SifakaDependencies(generator=config.model, critics=critics, validators=validators)

    def _get_cache_key(self, prompt: str, max_iterations: int) -> str:
        """Generate a cache key for the request."""
        content = f"{prompt}:{max_iterations}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[SifakaThought]:
        """Get result from cache if valid."""
        if cache_key not in self._cache:
            return None

        # Check TTL
        if cache_key in self._cache_timestamps:
            age = datetime.now() - self._cache_timestamps[cache_key]
            if age.total_seconds() > (self.config.cache_ttl_seconds if self.config else 3600):
                # Expired
                del self._cache[cache_key]
                del self._cache_timestamps[cache_key]
                return None

        return self._cache[cache_key]

    def _store_in_cache(self, cache_key: str, thought: SifakaThought) -> None:
        """Store result in cache."""
        # Enforce cache size limit
        max_size = self.config.cache_size if self.config else 1000
        if len(self._cache) >= max_size:
            # Remove oldest entry
            oldest_key = min(self._cache_timestamps.keys(), key=lambda k: self._cache_timestamps[k])
            del self._cache[oldest_key]
            del self._cache_timestamps[oldest_key]

        self._cache[cache_key] = thought
        self._cache_timestamps[cache_key] = datetime.now()

    async def think(
        self,
        prompt: str,
        max_iterations: Optional[int] = None,
        user_id: str = None,
        session_id: str = None,
    ) -> SifakaThought:
        """Process a single thought through the Sifaka workflow."""
        # Start timing if enabled
        start_time = time.time() if (self.config and self.config.enable_timing) else None
        request_id = str(uuid.uuid4())

        try:
            # Validate inputs
            prompt = validate_prompt(prompt)
            max_iterations = validate_max_iterations(max_iterations)

            # Determine max_iterations from config or parameter
            if max_iterations is None:
                max_iterations = self.config.max_iterations if self.config else 3

            # Logging if enabled
            if self.config and self.config.enable_logging:
                content_preview = (
                    f" - Content: {prompt[:100]}..." if self.config.log_content else ""
                )
                logger.info(
                    f"Starting thought processing for request {request_id}{content_preview}"
                )

            # Check cache if enabled
            cache_key = None
            if self.config and self.config.enable_caching:
                cache_key = self._get_cache_key(prompt, max_iterations)
                cached_result = self._get_from_cache(cache_key)
                if cached_result:
                    if self.config.enable_logging:
                        logger.info(f"Cache hit for request {request_id}")
                    return cached_result

            # Create initial thought
            thought = SifakaThought(prompt=prompt, max_iterations=max_iterations)

            logger.log_thought_event(
                "thought_created",
                thought.id,
                extra={
                    "prompt_length": len(prompt),
                    "max_iterations": max_iterations,
                    "request_id": request_id,
                },
            )

            # Run the graph starting with generation
            from sifaka.graph.nodes import GenerateNode

            with logger.performance_timer("graph_execution", thought_id=thought.id):
                result = await self.graph.run(
                    GenerateNode(), state=thought, deps=self.deps, persistence=self.persistence
                )

            # Extract the final thought from the GraphRunResult
            final_thought = result.state

            # Store in cache if enabled
            if self.config and self.config.enable_caching and cache_key:
                self._store_in_cache(cache_key, final_thought)

            # Record timing if enabled
            if start_time and self.config and self.config.enable_timing:
                duration = time.time() - start_time
                self._timing_data.append(
                    {
                        "request_id": request_id,
                        "duration_seconds": duration,
                        "iterations": final_thought.iteration,
                        "timestamp": datetime.now(),
                    }
                )
                if self.config.enable_logging:
                    logger.info(
                        f"Request {request_id} completed in {duration:.2f}s with {final_thought.iteration} iterations"
                    )

            logger.log_thought_event(
                "thought_completed",
                final_thought.id,
                iteration=final_thought.iteration,
                extra={
                    "final_iteration": final_thought.iteration,
                    "techniques_applied": final_thought.techniques_applied,
                    "text_length": (
                        len(final_thought.current_text) if final_thought.current_text else 0
                    ),
                    "is_finalized": final_thought.final_text is not None,
                    "request_id": request_id,
                },
            )

            return final_thought
        except Exception as e:
            if self.config and self.config.enable_logging:
                logger.error(f"Error in request {request_id}: {e}")

            logger.error(
                "Failed to process thought",
                extra={
                    "prompt_preview": prompt[:100],
                    "max_iterations": max_iterations,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "request_id": request_id,
                },
                exc_info=True,
            )
            raise GraphExecutionError(
                f"Failed to process thought: {str(e)}",
                execution_stage="graph_execution",
                context={
                    "prompt": prompt[:100],
                    "max_iterations": max_iterations,
                    "request_id": request_id,
                },
            ) from e

    def get_timing_stats(self) -> Dict[str, Any]:
        """Get timing statistics if timing is enabled."""
        if not self._timing_data:
            return {"message": "No timing data available. Enable timing in config."}

        durations = [d["duration_seconds"] for d in self._timing_data]
        iterations = [d["iterations"] for d in self._timing_data]

        return {
            "total_requests": len(self._timing_data),
            "avg_duration_seconds": sum(durations) / len(durations),
            "min_duration_seconds": min(durations),
            "max_duration_seconds": max(durations),
            "avg_iterations": sum(iterations) / len(iterations),
            "total_duration_seconds": sum(durations),
        }

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics if caching is enabled."""
        if not self.config or not self.config.enable_caching:
            return {"message": "Caching not enabled"}

        return {
            "cache_size": len(self._cache),
            "max_cache_size": self.config.cache_size,
            "cache_ttl_seconds": self.config.cache_ttl_seconds,
        }

    # Keep other methods from original engine...
    async def continue_thought(
        self, parent_thought: SifakaThought, new_prompt: str, max_iterations: int = 3
    ) -> SifakaThought:
        """Continue a conversation with a new thought connected to a parent."""
        # Validate inputs
        if not isinstance(parent_thought, SifakaThought):
            raise GraphExecutionError(
                f"parent_thought must be a SifakaThought, got {type(parent_thought).__name__}",
                execution_stage="input_validation",
                context={"parent_thought_type": type(parent_thought).__name__},
                suggestions=[
                    "Ensure parent_thought is a SifakaThought instance",
                    "Use engine.think() to create the initial thought first",
                ],
            )

        new_prompt = validate_prompt(new_prompt)
        max_iterations = validate_max_iterations(max_iterations)

        try:
            # Create new thought connected to parent
            new_thought = SifakaThought(prompt=new_prompt, max_iterations=max_iterations)
            new_thought.connect_to(parent_thought)

            # Run the graph for the new thought
            from sifaka.graph.nodes import GenerateNode

            result = await self.graph.run(
                GenerateNode(), state=new_thought, deps=self.deps, persistence=self.persistence
            )

            # Extract the final thought from the GraphRunResult
            return result.state
        except Exception as e:
            raise GraphExecutionError(
                f"Failed to continue thought: {str(e)}",
                execution_stage="graph_execution",
                context={
                    "new_prompt": new_prompt[:100],
                    "max_iterations": max_iterations,
                    "parent_id": parent_thought.id,
                },
            ) from e
