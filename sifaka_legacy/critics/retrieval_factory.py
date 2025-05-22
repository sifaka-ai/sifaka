"""
Factory functions for creating retrieval-enhanced critics.

This module provides factory functions for creating retrieval-enhanced versions of various critics.
"""

import logging
from typing import Any, List, Optional, cast

from sifaka.critics.base import Critic
from sifaka.critics.constitutional import create_constitutional_critic
from sifaka.critics.n_critics import create_n_critics_critic
from sifaka.critics.prompt import create_prompt_critic
from sifaka.critics.reflexion import create_reflexion_critic
from sifaka.critics.retrieval_enhanced import enhance_critic_with_retrieval
from sifaka.critics.self_refine import create_self_refine_critic
from sifaka.models.base import Model
from sifaka.retrievers.augmenter import RetrievalAugmenter

logger = logging.getLogger(__name__)


def create_retrieval_enhanced_constitutional_critic(
    model: Model,
    retrieval_augmenter: RetrievalAugmenter,
    principles: Optional[List[str]] = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    include_passages_in_critique: bool = True,
    include_passages_in_improve: bool = True,
    max_passages: int = 5,
    **options: Any,
) -> Critic:
    """Create a retrieval-enhanced constitutional critic.

    Args:
        model: The model to use for critiquing and improving text.
        retrieval_augmenter: The retrieval augmenter to use for retrieving passages.
        principles: Optional list of constitutional principles to use.
        system_prompt: The system prompt to use for the model.
        temperature: The temperature to use for the model.
        include_passages_in_critique: Whether to include retrieved passages in the critique.
        include_passages_in_improve: Whether to include retrieved passages in the improve method.
        max_passages: Maximum number of passages to retrieve.
        **options: Additional options to pass to the critic.

    Returns:
        A retrieval-enhanced constitutional critic.

    Raises:
        ImproverError: If the model or retrieval augmenter is not provided.
    """
    # Create a base constitutional critic
    base_critic = create_constitutional_critic(
        model=model,
        principles=principles,
        system_prompt=system_prompt,
        temperature=temperature,
        **options,
    )

    # Enhance it with retrieval
    return enhance_critic_with_retrieval(
        critic=cast(Critic, base_critic),
        retrieval_augmenter=retrieval_augmenter,
        include_passages_in_critique=include_passages_in_critique,
        include_passages_in_improve=include_passages_in_improve,
        max_passages=max_passages,
    )


def create_retrieval_enhanced_n_critics_critic(
    model: Model,
    retrieval_augmenter: RetrievalAugmenter,
    num_critics: int = 3,
    max_refinement_iterations: int = 2,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    include_passages_in_critique: bool = True,
    include_passages_in_improve: bool = True,
    max_passages: int = 5,
    **options: Any,
) -> Critic:
    """Create a retrieval-enhanced N-Critics critic.

    Args:
        model: The model to use for critiquing and improving text.
        retrieval_augmenter: The retrieval augmenter to use for retrieving passages.
        num_critics: Number of critics to use.
        max_refinement_iterations: Maximum number of refinement iterations.
        system_prompt: The system prompt to use for the model.
        temperature: The temperature to use for the model.
        include_passages_in_critique: Whether to include retrieved passages in the critique.
        include_passages_in_improve: Whether to include retrieved passages in the improve method.
        max_passages: Maximum number of passages to retrieve.
        **options: Additional options to pass to the critic.

    Returns:
        A retrieval-enhanced N-Critics critic.

    Raises:
        ImproverError: If the model or retrieval augmenter is not provided.
    """
    # Create a base N-Critics critic
    base_critic = create_n_critics_critic(
        model=model,
        num_critics=num_critics,
        max_refinement_iterations=max_refinement_iterations,
        system_prompt=system_prompt,
        temperature=temperature,
        **options,
    )

    # Enhance it with retrieval
    return enhance_critic_with_retrieval(
        critic=cast(Critic, base_critic),
        retrieval_augmenter=retrieval_augmenter,
        include_passages_in_critique=include_passages_in_critique,
        include_passages_in_improve=include_passages_in_improve,
        max_passages=max_passages,
    )


def create_retrieval_enhanced_reflexion_critic(
    model: Model,
    retrieval_augmenter: RetrievalAugmenter,
    reflection_rounds: int = 2,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    include_passages_in_critique: bool = True,
    include_passages_in_improve: bool = True,
    max_passages: int = 5,
    **options: Any,
) -> Critic:
    """Create a retrieval-enhanced Reflexion critic.

    Args:
        model: The model to use for critiquing and improving text.
        retrieval_augmenter: The retrieval augmenter to use for retrieving passages.
        reflection_rounds: Number of reflection rounds to perform.
        system_prompt: The system prompt to use for the model.
        temperature: The temperature to use for the model.
        include_passages_in_critique: Whether to include retrieved passages in the critique.
        include_passages_in_improve: Whether to include retrieved passages in the improve method.
        max_passages: Maximum number of passages to retrieve.
        **options: Additional options to pass to the critic.

    Returns:
        A retrieval-enhanced Reflexion critic.

    Raises:
        ImproverError: If the model or retrieval augmenter is not provided.
    """
    # Create a base Reflexion critic
    base_critic = create_reflexion_critic(
        model=model,
        reflection_rounds=reflection_rounds,
        system_prompt=system_prompt,
        temperature=temperature,
        **options,
    )

    # Enhance it with retrieval
    return enhance_critic_with_retrieval(
        critic=cast(Critic, base_critic),
        retrieval_augmenter=retrieval_augmenter,
        include_passages_in_critique=include_passages_in_critique,
        include_passages_in_improve=include_passages_in_improve,
        max_passages=max_passages,
    )


def create_retrieval_enhanced_self_refine_critic(
    model: Model,
    retrieval_augmenter: RetrievalAugmenter,
    max_iterations: int = 3,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    include_passages_in_critique: bool = True,
    include_passages_in_improve: bool = True,
    max_passages: int = 5,
    **options: Any,
) -> Critic:
    """Create a retrieval-enhanced Self-Refine critic.

    Args:
        model: The model to use for critiquing and improving text.
        retrieval_augmenter: The retrieval augmenter to use for retrieving passages.
        max_iterations: Maximum number of refinement iterations.
        system_prompt: The system prompt to use for the model.
        temperature: The temperature to use for the model.
        include_passages_in_critique: Whether to include retrieved passages in the critique.
        include_passages_in_improve: Whether to include retrieved passages in the improve method.
        max_passages: Maximum number of passages to retrieve.
        **options: Additional options to pass to the critic.

    Returns:
        A retrieval-enhanced Self-Refine critic.

    Raises:
        ImproverError: If the model or retrieval augmenter is not provided.
    """
    # Create a base Self-Refine critic
    base_critic = create_self_refine_critic(
        model=model,
        max_iterations=max_iterations,
        system_prompt=system_prompt,
        temperature=temperature,
        **options,
    )

    # Enhance it with retrieval
    return enhance_critic_with_retrieval(
        critic=cast(Critic, base_critic),
        retrieval_augmenter=retrieval_augmenter,
        include_passages_in_critique=include_passages_in_critique,
        include_passages_in_improve=include_passages_in_improve,
        max_passages=max_passages,
    )


def create_retrieval_enhanced_prompt_critic(
    model: Model,
    retrieval_augmenter: RetrievalAugmenter,
    critique_prompt: Optional[str] = None,
    improve_prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    include_passages_in_critique: bool = True,
    include_passages_in_improve: bool = True,
    max_passages: int = 5,
    **options: Any,
) -> Critic:
    """Create a retrieval-enhanced Prompt critic.

    Args:
        model: The model to use for critiquing and improving text.
        retrieval_augmenter: The retrieval augmenter to use for retrieving passages.
        critique_prompt: Optional custom prompt for critique.
        improve_prompt: Optional custom prompt for improvement.
        system_prompt: The system prompt to use for the model.
        temperature: The temperature to use for the model.
        include_passages_in_critique: Whether to include retrieved passages in the critique.
        include_passages_in_improve: Whether to include retrieved passages in the improve method.
        max_passages: Maximum number of passages to retrieve.
        **options: Additional options to pass to the critic.

    Returns:
        A retrieval-enhanced Prompt critic.

    Raises:
        ImproverError: If the model or retrieval augmenter is not provided.
    """
    # Create a base Prompt critic
    base_critic = create_prompt_critic(
        model=model,
        critique_prompt=critique_prompt,
        improve_prompt=improve_prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        **options,
    )

    # Enhance it with retrieval
    return enhance_critic_with_retrieval(
        critic=cast(Critic, base_critic),
        retrieval_augmenter=retrieval_augmenter,
        include_passages_in_critique=include_passages_in_critique,
        include_passages_in_improve=include_passages_in_improve,
        max_passages=max_passages,
    )
