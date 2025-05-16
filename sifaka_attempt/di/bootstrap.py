"""
Bootstrap module for initializing the dependency injection container.

This module provides functions to initialize the DI container with core components
of the Sifaka library.
"""

import logging
from typing import Dict, Any, Optional, List, Type

from sifaka.di import (
    register,
    register_factory,
    get_container,
    DependencyScope,
)
from sifaka.models import ModelProvider, create_model
from sifaka.validators import ValidatorProtocol
from sifaka.critics import CriticProtocol

logger = logging.getLogger(__name__)


def initialize_di_container() -> None:
    """
    Initialize the DI container with core components.

    This function registers core factories and utilities.
    """
    logger.info("Initializing DI container")

    # Clear existing dependencies to ensure a clean state
    get_container().clear()

    # Register model factories
    register_factory("model.openai", lambda: create_model("openai"), DependencyScope.SINGLETON)
    register_factory(
        "model.anthropic", lambda: create_model("anthropic"), DependencyScope.SINGLETON
    )

    # Register a model factory function
    def model_factory(model_type: str, **kwargs) -> ModelProvider:
        return create_model(model_type, **kwargs)

    register("model.factory", model_factory, DependencyScope.SINGLETON)

    logger.info("DI container initialized")


def register_model(name: str, model_type: str, **kwargs) -> None:
    """
    Register a model with the DI container.

    Args:
        name: The name to register the model under
        model_type: The type of model to create
        **kwargs: Additional parameters for the model constructor
    """

    def factory():
        return create_model(model_type, **kwargs)

    register_factory(name, factory, DependencyScope.SINGLETON)
    logger.info(f"Registered model {name} of type {model_type}")


def register_validator(name: str, validator: ValidatorProtocol) -> None:
    """
    Register a validator with the DI container.

    Args:
        name: The name to register the validator under
        validator: The validator instance
    """
    register(name, validator, DependencyScope.SINGLETON)
    logger.info(f"Registered validator {name}")


def register_critic(name: str, critic: CriticProtocol) -> None:
    """
    Register a critic with the DI container.

    Args:
        name: The name to register the critic under
        critic: The critic instance
    """
    register(name, critic, DependencyScope.SINGLETON)
    logger.info(f"Registered critic {name}")


def register_component_factories() -> None:
    """
    Register component factories with the DI container.

    This function registers factories for creating various components
    like validators and critics.
    """
    # Register validator factories
    from sifaka.validators.length import LengthValidator
    from sifaka.validators.content import ContentValidator
    from sifaka.validators.toxicity import ToxicityValidator
    from sifaka.validators.readability import ReadabilityValidator
    from sifaka.validators.grammar import GrammarValidator
    from sifaka.validators.similarity import SimilarityValidator
    from sifaka.validators.custom import CustomValidator
    from sifaka.validators.sentiment import SentimentValidator
    from sifaka.validators.spam import SpamValidator
    from sifaka.validators.bias import BiasValidator

    register_factory("validator.length", lambda **kwargs: LengthValidator(**kwargs))
    register_factory("validator.content", lambda **kwargs: ContentValidator(**kwargs))
    register_factory("validator.toxicity", lambda **kwargs: ToxicityValidator(**kwargs))
    register_factory("validator.readability", lambda **kwargs: ReadabilityValidator(**kwargs))
    register_factory("validator.grammar", lambda **kwargs: GrammarValidator(**kwargs))
    register_factory("validator.similarity", lambda **kwargs: SimilarityValidator(**kwargs))
    register_factory("validator.custom", lambda **kwargs: CustomValidator(**kwargs))
    register_factory("validator.sentiment", lambda **kwargs: SentimentValidator(**kwargs))
    register_factory("validator.spam", lambda **kwargs: SpamValidator(**kwargs))
    register_factory("validator.bias", lambda **kwargs: BiasValidator(**kwargs))

    # Register critic factories
    from sifaka.critics.prompt import PromptCritic
    from sifaka.critics.reflexion import ReflexionCritic
    from sifaka.critics.constitutional import ConstitutionalCritic
    from sifaka.critics.self_rag import SelfRAGCritic
    from sifaka.critics.self_refine import SelfRefineCritic
    from sifaka.critics.lac import LACCritic

    register_factory("critic.prompt", lambda **kwargs: PromptCritic(**kwargs))
    register_factory("critic.reflexion", lambda **kwargs: ReflexionCritic(**kwargs))
    register_factory("critic.constitutional", lambda **kwargs: ConstitutionalCritic(**kwargs))
    register_factory("critic.self_rag", lambda **kwargs: SelfRAGCritic(**kwargs))
    register_factory("critic.self_refine", lambda **kwargs: SelfRefineCritic(**kwargs))
    register_factory("critic.lac", lambda **kwargs: LACCritic(**kwargs))

    # Register classifier factories
    from sifaka.classifiers.toxicity import ToxicityClassifier
    from sifaka.classifiers.sentiment import SentimentClassifier
    from sifaka.classifiers.spam import SpamClassifier
    from sifaka.classifiers.bias import BiasClassifier

    register_factory("classifier.toxicity", lambda **kwargs: ToxicityClassifier(**kwargs))
    register_factory("classifier.sentiment", lambda **kwargs: SentimentClassifier(**kwargs))
    register_factory("classifier.spam", lambda **kwargs: SpamClassifier(**kwargs))
    register_factory("classifier.bias", lambda **kwargs: BiasClassifier(**kwargs))

    logger.info("Component factories registered")
