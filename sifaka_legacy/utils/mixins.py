"""
Utility mixins for Sifaka.

This module provides utility mixin classes for use across the Sifaka codebase.
"""


class InitializeStateMixin:
    """
    A mixin that provides an empty _initialize_state implementation.

    This mixin is used to provide a base implementation of _initialize_state
    for classes that need to call super()._initialize_state() but don't have
    a parent class that implements it.

    Classes can inherit from this mixin to avoid mypy errors about
    _initialize_state being undefined in the superclass.
    """

    def _initialize_state(self) -> None:
        """
        Initialize component state.

        This is a base implementation that does nothing and can be called
        by subclasses with super()._initialize_state().
        """
        pass
