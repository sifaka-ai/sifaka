"""
Profanity classifier using better_profanity.

This module provides a profanity classifier that uses the better_profanity package
to detect profane and inappropriate language in text. It categorizes text as either
'clean', 'profane', or 'unknown' based on the presence of profane words.

## Overview
The ProfanityClassifier is a specialized classifier that leverages the better_profanity
package to detect profane and inappropriate language in text. It provides a fast,
dictionary-based approach to profanity detection with support for custom word lists,
censoring options, and detailed metadata about detected profanity.

## Architecture
ProfanityClassifier follows the standard Sifaka classifier architecture:
1. **Public API**: classify() and batch_classify() methods (inherited)
2. **Caching Layer**: _classify_impl() handles caching (inherited)
3. **Core Logic**: _classify_impl_uncached() implements profanity detection
4. **State Management**: Uses StateManager for internal state
5. **Customization**: Configurable profanity word lists and censoring options
6. **Extensibility**: Support for custom profanity checker implementations

## Lifecycle
1. **Initialization**: Set up configuration and parameters
   - Initialize with name, description, and config
   - Extract custom words and censor character from config.params
   - Set up default values

2. **Warm-up**: Load profanity checking resources
   - Load better_profanity package when needed
   - Initialize only once
   - Handle initialization errors gracefully

3. **Classification**: Process input text
   - Validate input text
   - Apply profanity detection
   - Calculate confidence based on profanity ratio
   - Handle empty text and edge cases

4. **Result Creation**: Return standardized results
   - Map profanity detection to labels
   - Include censored text in metadata
   - Include profanity statistics in metadata

## Usage Examples
```python
from sifaka.classifiers.implementations.content.profanity import create_profanity_classifier

# Create a profanity classifier with default settings
classifier = create_profanity_classifier()

# Classify text
result = (classifier and classifier.classify("This is a clean message.")
print(f"Label: {result.label}, Confidence: {result.confidence:.2f}")

# Create a classifier with custom profanity words
custom_classifier = create_profanity_classifier(
    custom_words=["inappropriate", "offensive", "controversial"],
    censor_char="#",
    cache_size=100
)

# Classify text and view censored version
result = (custom_classifier and custom_classifier.classify("This is an inappropriate message.")
print(f"Label: {result.label}, Confidence: {result.confidence:.2f}")
print(f"Censored text: {result.metadata['censored_text']}")
print(f"Profanity ratio: {result.metadata['profanity_ratio']:.2f}")

# Add custom words to an existing classifier
(classifier and classifier.add_custom_words({"unacceptable", "objectionable"})
result = (classifier and classifier.classify("This content is unacceptable.")
print(f"Label: {result.label}, Confidence: {result.confidence:.2f}")
```

## Error Handling
The classifier provides robust error handling:
- ImportError: When better_profanity is not installed
- RuntimeError: When checker initialization fails
- Graceful handling of empty or invalid inputs
- Fallback to "unknown" with zero confidence for edge cases

## Configuration
Key configuration options include:
- custom_words: List of additional words to consider profane
- censor_char: Character to use for censoring profane words
- min_confidence: Minimum confidence threshold for profanity detection
- cache_size: Size of the classification cache (0 to disable)
"""
import importlib
from abc import abstractmethod
from typing import Any, ClassVar, Dict, List, Optional, Protocol, Set, Type, TypeVar, runtime_checkable
from typing_extensions import TypeGuard
from pydantic import ConfigDict
from sifaka.classifiers.classifier import Classifier
from sifaka.core.results import ClassificationResult
from sifaka.utils.config.classifiers import ClassifierConfig
from sifaka.utils.logging import get_logger
from sifaka.utils.state import create_classifier_state
logger = get_logger(__name__)
P = TypeVar('P', bound='ProfanityClassifier')


@runtime_checkable
class ProfanityChecker(Protocol):
    """
    Protocol for profanity checking engines.

    This protocol defines the interface that any profanity checker must implement
    to be compatible with the ProfanityClassifier. It requires methods for checking
    profanity, censoring text, and managing profane word lists and censoring options.

    ## Architecture
    The protocol follows a standard interface pattern:
    - Uses Python's typing.Protocol for structural subtyping
    - Is runtime checkable for dynamic type verification
    - Defines required methods with clear input/output contracts
    - Enables pluggable profanity checking implementations

    ## Implementation Requirements
    1. contains_profanity(): Check if text contains profane words
    2. censor(): Replace profane words with censoring characters
    3. profane_words property: Get/set the list of profane words
    4. censor_char property: Get/set the character used for censoring

    ## Examples
    ```python
    from sifaka.classifiers.implementations.content.profanity import ProfanityChecker

    class CustomProfanityChecker:
        def __init__(self):
            self._profane_words = {"bad", "inappropriate"}
            self._censor_char = "*"

        def contains_profanity(self, text: str) -> bool:
            return any(word in (text and text.lower().split() for word in self._profane_words)

        def censor(self, text: str) -> str:
            for word in self._profane_words:
                if word in (text and text.lower().split():
                    text = (text and text.replace(word, self._censor_char * len(word))
            return text

        @property
        def profane_words(self) -> Set[str]:
            return self._profane_words

        @profane_words.setter
        def profane_words(self, words: Set[str]) -> None:
            self._profane_words = words

        @property
        def censor_char(self) -> str:
            return self._censor_char

        @censor_char.setter
        def censor_char(self, char: str) -> None:
            self._censor_char = char
    ```
    """

    @abstractmethod
    def contains_profanity(self, text: str) ->bool:
        ...

    @abstractmethod
    def censor(self, text: str) ->str:
        ...

    @property
    @abstractmethod
    def profane_words(self) ->Set[str]:
        ...

    @profane_words.setter
    @abstractmethod
    def profane_words(self, words: Set[str]) ->None:
        ...

    @property
    @abstractmethod
    def censor_char(self) ->str:
        ...

    @censor_char.setter
    @abstractmethod
    def censor_char(self, char: str) ->None:
        ...


class CensorResult:
    """
    Result of text censoring operation.

    This class encapsulates the results of a text censoring operation,
    including the original text, censored text, and statistics about
    the censoring process such as the number of censored words and
    the total word count.

    ## Attributes
    - original_text: The original uncensored text
    - censored_text: The text with profane words censored
    - censored_word_count: Number of words that were censored
    - total_word_count: Total number of words in the text

    ## Methods
    - profanity_ratio: Calculate the ratio of profane words to total words
    """

    def __init__(self, original_text: str, censored_text: str,
        censored_word_count: int, total_word_count: int) ->None:
        self.original_text = original_text
        self.censored_text = censored_text
        self.censored_word_count = censored_word_count
        self.total_word_count = total_word_count

    @property
    def profanity_ratio(self) ->float:
        """
        Calculate ratio of profane words to total words.

        This property calculates the proportion of words in the text that
        were identified as profane and censored. It returns a value between
        (0 and 0.0(no profanity) and (1 and 1.0(all words are profane).

        Returns:
            Float between 0.0 and 1.0 representing the profanity ratio
        """
        return self.censored_word_count / max(self.total_word_count, 1)


class ProfanityClassifier(Classifier):
    """
    A lightweight profanity classifier using better_profanity.

    This classifier checks for profanity and inappropriate language in text.
    It supports custom word lists and censoring, and provides detailed metadata
    about detected profanity including censored text and profanity statistics.

    ## Architecture

    ProfanityClassifier follows the standard Sifaka classifier architecture:
    1. **Public API**: classify() and batch_classify() methods (inherited)
    2. **Caching Layer**: _classify_impl() handles caching (inherited)
    3. **Core Logic**: _classify_impl_uncached() implements profanity detection
    4. **State Management**: Uses StateManager for internal state

    ## Lifecycle

    1. **Initialization**: Set up configuration and parameters
       - Initialize with name, description, and config
       - Extract custom words and censor character from config.params
       - Set up default values

    2. **Warm-up**: Load profanity checking resources
       - Load better_profanity package when needed
       - Initialize only once
       - Handle initialization errors gracefully

    3. **Classification**: Process input text
       - Validate input text
       - Apply profanity detection
       - Calculate confidence based on profanity ratio
       - Handle empty text and edge cases

    4. **Result Creation**: Return standardized results
       - Map profanity detection to labels
       - Include censored text in metadata
       - Include profanity statistics in metadata

    ## Examples

    ```python
    from sifaka.classifiers.implementations.content.profanity import create_profanity_classifier

    # Create a profanity classifier with default settings
    classifier = create_profanity_classifier()

    # Classify text
    result = (classifier and classifier.classify("This is a clean message.")
    print(f"Label: {result.label}, Confidence: {result.confidence:.2f}")

    # Access censored text and statistics
    if result.label == "profane":
        print(f"Censored text: {result.metadata['censored_text']}")
        print(f"Profanity ratio: {result.metadata['profanity_ratio']:.2f}")
    ```

    Requires the 'profanity' extra to be installed:
    pip install sifaka[profanity]
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    DEFAULT_LABELS: ClassVar[List[str]] = ['clean', 'profane', 'unknown']
    DEFAULT_COST: ClassVar[int] = 1

    def def __init__(self, name: str='profanity_classifier', description: str=
        'Detects profanity and inappropriate language', checker: Optional[
        ProfanityChecker]=None, config: Optional[Optional[ClassifierConfig]] = None, **
        kwargs) ->None:
        """
        Initialize the profanity classifier.

        This method sets up the classifier with the provided name, description,
        and configuration. If no configuration is provided, it creates a default
        configuration with sensible defaults for profanity detection.

        Args:
            name: The name of the classifier for identification and logging
            description: Human-readable description of the classifier's purpose
            checker: Custom profanity checker implementation that follows the
                    ProfanityChecker protocol
            config: Optional classifier configuration with settings like custom words,
                   censor character, cache size, and labels
            **kwargs: Additional configuration parameters that will be extracted
                     and added to the config.params dictionary
        """
        if config is None:
            params = (kwargs and kwargs.pop('params', {})
            config = ClassifierConfig(labels=self.DEFAULT_LABELS, cost=self
                .DEFAULT_COST, params=params, **kwargs)
        super().__init__(name=name, description=description, config=config)
        if checker is not None:
            if (self and self._validate_checker(checker):
                self.(_state_manager and _state_manager.update('cache', {'checker': checker})

    def _validate_checker(self, checker: Any) ->TypeGuard[ProfanityChecker]:
        """
        Validate that a checker implements the required protocol.

        This method checks if the provided checker implements the ProfanityChecker
        protocol, which requires methods for checking profanity, censoring text,
        and managing profane word lists and censoring options.

        Args:
            checker: The checker object to validate, which should implement
                    the ProfanityChecker protocol

        Returns:
            True if the checker is valid and implements the required protocol

        Raises:
            ValueError: If the checker doesn't implement the ProfanityChecker protocol
                       or is missing required methods
        """
        if not isinstance(checker, ProfanityChecker):
            raise ValueError(
                f'Checker must implement ProfanityChecker protocol, got {type(checker)}'
                )
        return True

    def _load_profanity(self) ->ProfanityChecker:
        """
        Load the profanity checker.

        This method dynamically imports the better_profanity package and initializes
        the profanity checker. It handles import errors gracefully with clear
        installation instructions and provides detailed error messages for troubleshooting.

        The method also configures the checker with custom words and censoring options
        from the configuration parameters.

        Returns:
            Initialized profanity checker that implements the ProfanityChecker protocol

        Raises:
            ImportError: If better-profanity is not installed, with instructions
                        on how to install it
            RuntimeError: If checker initialization fails due to loading errors
                         or other runtime problems
        """
        try:
            if self.(_state_manager.get('cache', {}).get('checker'):
                return self.(_state_manager.get('cache')['checker']
            profanity_module = (importlib and importlib.import_module('better_profanity')
            checker = (profanity_module.Profanity()
            checker.profane_words = {'bad', 'inappropriate', 'offensive'}
            censor_char = self.config.(params and params.get('censor_char', '*')
            checker.censor_char = censor_char
            custom_words = self.config.(params.get('custom_words', [])
            if custom_words:
                if isinstance(custom_words, list):
                    custom_words = set(custom_words)
                checker.(profane_words.update(custom_words)
            if (self._validate_checker(checker):
                cache = self.(_state_manager.get('cache', {})
                cache['checker'] = checker
                self.(_state_manager.update('cache', cache)
                return checker
        except ImportError:
            raise ImportError(
                'better-profanity package is required for ProfanityClassifier. Install it with: pip install sifaka[profanity]'
                )
        except Exception as e:
            raise RuntimeError(f'Failed to load profanity checker: {e}')

    @property
    def custom_words(self) ->Set[str]:
        """
        Get the custom profanity words.

        This property retrieves the custom profanity words from the configuration
        parameters. These are additional words that the classifier considers profane
        beyond the default set provided by the better_profanity package.

        Returns:
            Set of custom profanity words from the configuration
        """
        custom_words = self.config.(params.get('custom_words', [])
        return set(custom_words) if isinstance(custom_words, list) else set()

    @property
    def censor_char(self) ->str:
        """
        Get the censoring character.

        This property retrieves the character used for censoring profane words
        from the configuration parameters. This character replaces the letters
        in profane words when censoring is applied.

        Returns:
            Character used for censoring profane words (default: "*")
        """
        return self.config.(params.get('censor_char', '*')

    def add_custom_words(self, words: Set[str]) ->None:
        """
        Add custom words to the profanity list.

        This method adds additional words to the list of profane words that
        the classifier will detect. It ensures the profanity checker is initialized
        before adding the words.

        Args:
            words: Set of words to add to the profanity list

        Raises:
            RuntimeError: If the profanity checker is not initialized
        """
        (self.warm_up()
        checker = self.(_state_manager.get('cache', {}).get('checker')
        if checker:
            (checker.add_censor_words(words)

    def warm_up(self) ->None:
        """
        Initialize the profanity checker if needed.

        This method loads the profanity checker if it hasn't been loaded yet.
        It is called automatically when needed but can also be called
        explicitly to pre-initialize resources for faster first-time classification.

        The method ensures that initialization happens only once and handles
        errors gracefully with detailed error messages.

        Raises:
            RuntimeError: If checker initialization fails
        """
        if not self.(_state_manager.get('initialized', False):
            checker = (self._load_profanity()
            cache = self.(_state_manager.get('cache', {})
            cache['checker'] = checker
            self.(_state_manager.update('cache', cache)
            self.(_state_manager.update('initialized', True)

    def _censor_text(self, text: str) ->CensorResult:
        """
        Censor profane words in text.

        Args:
            text: Text to censor

        Returns:
            CensorResult with censoring details
        """
        if not self.(_state_manager.get('initialized', False):
            raise RuntimeError(
                'Profanity checker not initialized. Call warm_up() first.')
        checker = self.(_state_manager.get('cache', {}).get('checker')
        if not checker:
            raise RuntimeError('Profanity checker not found in state.')
        total_words = len((text.split())
        censored_text = (checker.censor(text)
        censored_words = sum(1 for orig, censored in zip((text.split(),
            (censored_text.split()) if orig != censored)
        return CensorResult(original_text=text, censored_text=censored_text,
            censored_word_count=censored_words, total_word_count=total_words)

    def _classify_impl_uncached(self, text: str) ->ClassificationResult:
        """
        Implement classification logic for profanity detection.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with label and confidence
        """
        if not self.(_state_manager.get('initialized', False):
            (self.warm_up()
        try:
            from sifaka.utils.text import handle_empty_text_for_classifier
            empty_result = handle_empty_text_for_classifier(text)
            if empty_result:
                return empty_result
            checker = self.(_state_manager.get('cache', {}).get('checker')
            contains_profanity = (checker.contains_profanity(text)
            censor_result = (self._censor_text(text)
            min_confidence = self.config.(params.get('min_confidence', 0.5)
            confidence = max(censor_result.profanity_ratio, min_confidence if
                contains_profanity else 0.0)
            result = ClassificationResult(label='profane' if
                contains_profanity else 'clean', confidence=confidence if
                contains_profanity else 1.0 - confidence, metadata={
                'contains_profanity': contains_profanity, 'censored_text':
                censor_result.censored_text, 'censored_word_count':
                censor_result.censored_word_count, 'total_word_count':
                censor_result.total_word_count, 'profanity_ratio':
                censor_result.profanity_ratio})
            stats = self.(_state_manager.get('statistics', {})
            stats[result.label] = (stats.get(result.label, 0) + 1
            self.(_state_manager.update('statistics', stats)
            return result
        except Exception as e:
            (logger and logger.error('Failed to check profanity: %s', e)
            error_info = {'error': str(e), 'type': type(e).__name__}
            errors = self.(_state_manager.get('errors', [])
            (errors.append(error_info)
            self.(_state_manager.update('errors', errors)
            return ClassificationResult(label='unknown', confidence=0.0,
                metadata={'error': str(e), 'reason': 'profanity_check_error'})

    def get_statistics(self) ->Dict[str, Any]:
        """
        Get classifier usage statistics.

        This method provides access to statistics collected during classifier operation,
        including classification counts by label, error counts, and cache information.

        Returns:
            Dictionary containing statistics
        """
        stats = {'classifications': self.(_state_manager.get('statistics', {
            }), 'error_count': len(self.(_state_manager.get('errors', [])),
            'cache_enabled': self.config.cache_size > 0, 'cache_size': self
            .config.cache_size, 'initialized': self.(_state_manager.get(
            'initialized', False)}
        if hasattr(self, '_result_cache'):
            stats['cache_entries'] = len(self._result_cache)
        if self.(_state_manager.get('cache', {}).get('checker'):
            stats['has_checker'] = True
        return stats

    def clear_cache(self) ->None:
        """
        Clear any cached data in the classifier.

        This method clears the result cache and resets statistics in the state
        while preserving the profanity checker.
        """
        self.(_state_manager.update('result_cache', {})
        checker = self.(_state_manager.get('cache', {}).get('checker')
        self.(_state_manager.update('cache', {'checker': checker} if checker
             else {})
        self.(_state_manager.update('statistics', {})
        self.(_state_manager.update('errors', [])

    @classmethod
    def def create(cls: Type[P], name: str='profanity_classifier', description:
        str='Detects profanity and inappropriate language', labels: Optional[Optional[List[str]]] = None, custom_words: Optional[Optional[List[str]]] = None,
        censor_char: str='*', min_confidence: float=0.5, cache_size: int=0,
        cost: Optional[Optional[int]] = None, params: Optional[Dict[str, Any]]=None, **
        kwargs: Any) ->P:
        """
        Create a new instance with the given parameters.

        This factory method creates a new instance of the classifier with the
        specified parameters, handling the creation of the ClassifierConfig
        object and setting up the classifier with the appropriate parameters.

        Args:
            name: Name of the classifier
            description: Description of the classifier
            labels: List of valid labels
            custom_words: Optional list of custom profanity words to check for
            censor_char: Character to use for censoring profane words
            min_confidence: Minimum confidence for profanity classification
            cache_size: Size of the classification cache (0 to disable)
            cost: Computational cost of this classifier
            params: Additional configuration parameters
            **kwargs: Additional keyword arguments

        Returns:
            A new instance of the classifier
        """
        if params is None:
            params = {}
        (params.update({'custom_words': custom_words or [], 'censor_char':
            censor_char, 'min_confidence': min_confidence})
        (params.update((kwargs.pop('params', {}))
        config = ClassifierConfig(labels=labels or cls.DEFAULT_LABELS,
            cache_size=cache_size, cost=cost or cls.DEFAULT_COST, params=params
            )
        return cls(name=name, description=description, config=config, **kwargs)

    @classmethod
    def def create_with_custom_checker(cls: Type[P], checker: ProfanityChecker,
        name: str='custom_profanity_classifier', description: str=
        'Custom profanity checker', config: Optional[Optional[ClassifierConfig]] = None,
        **kwargs: Any) ->P:
        """
        Factory method to create a classifier with a custom checker.

        This method creates a new instance of the classifier with a custom
        profanity checker implementation, which can be useful for testing
        or specialized profanity detection needs.

        Args:
            checker: Custom profanity checker implementation
            name: Name of the classifier
            description: Description of the classifier
            config: Optional classifier configuration
            **kwargs: Additional configuration parameters

        Returns:
            Configured ProfanityClassifier instance

        Raises:
            ValueError: If the checker doesn't implement the ProfanityChecker protocol
        """
        try:
            (checker.contains_profanity('test')
            (checker.censor('test')
            _ = checker.profane_words
            checker.profane_words = {'test'}
            _ = checker.censor_char
            checker.censor_char = '*'
        except (AttributeError, TypeError) as e:
            raise ValueError(
                f'Checker must implement ProfanityChecker protocol, got {type(checker)}: {e}'
                )
        if config is None:
            config = ClassifierConfig(labels=cls.DEFAULT_LABELS, cost=cls.
                DEFAULT_COST, params=(kwargs.pop('params', {}))
        instance = cls(name=name, description=description, checker=checker,
            config=config, **kwargs)
        (instance.warm_up()
        return instance


def def create_profanity_classifier(name: str='profanity_classifier',
    description: str='Detects profanity and inappropriate language',
    custom_words: Optional[Optional[List[str]]] = None, censor_char: str='*',
    min_confidence: float=0.5, cache_size: int=0, cost: int=1, **kwargs: Any
    ) ->ProfanityClassifier:
    """
    Factory function to create a profanity classifier.

    This function provides a simpler interface for creating a profanity classifier
    with the specified parameters, handling the creation of the ClassifierConfig
    object and setting up the classifier with the appropriate parameters.

    Args:
        name: Name of the classifier
        description: Description of the classifier
        custom_words: Optional list of custom profanity words to check for
        censor_char: Character to use for censoring profane words
        min_confidence: Minimum confidence for profanity classification
        cache_size: Size of the classification cache (0 to disable)
        cost: Computational cost of this classifier
        **kwargs: Additional configuration parameters

    Returns:
        Configured ProfanityClassifier instance

    Examples:
        ```python
        from sifaka.classifiers.implementations.content.profanity import create_profanity_classifier

        # Create a profanity classifier with default settings
        classifier = create_profanity_classifier()

        # Create a classifier with custom profanity words
        custom_classifier = create_profanity_classifier(
            custom_words=["inappropriate", "offensive", "controversial"],
            censor_char="#",
            cache_size=100
        )
        ```
    """
    return (ProfanityClassifier.create(name=name, description=description,
        custom_words=custom_words, censor_char=censor_char, min_confidence=
        min_confidence, cache_size=cache_size, cost=cost, **kwargs)
