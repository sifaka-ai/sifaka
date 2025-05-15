from typing import Any, Dict, List, Optional, TypeVar, Generic, Type
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

class State(BaseModel):
    data: Dict[str, Any]
    metadata: Dict[str, Any]

class StateManager:
    _state: State
    _history: List[State]

    def __init__(self) -> None: ...
    def update(self, key: str, value: Any) -> None: ...
    def rollback(self) -> None: ...
    def get(self, key: str, default: Optional[Any] = None) -> Any: ...
    def set_metadata(self, key: str, value: Any) -> None: ...
    def get_metadata(self, key: str, default: Optional[Any] = None) -> Any: ...
    def reset(self) -> None: ...

class ComponentState(BaseModel):
    initialized: bool
    error: Optional[str]

class ClassifierState(ComponentState):
    model: Optional[Any]
    vectorizer: Optional[Any]
    pipeline: Optional[Any]
    feature_names: Dict[str, Any]
    cache: Dict[str, Any]
    dependencies_loaded: bool

class RuleState(ComponentState):
    validator: Optional[Any]
    handler: Optional[Any]
    cache: Dict[str, Any]
    compiled_patterns: Dict[str, Any]

class CriticState(ComponentState):
    model: Optional[Any]
    prompt_manager: Optional[Any]
    response_parser: Optional[Any]
    memory_manager: Optional[Any]
    cache: Dict[str, Any]

class ModelState(ComponentState):
    client: Optional[Any]
    token_counter: Optional[Any]
    tracer: Optional[Any]
    cache: Dict[str, Any]

class ChainState(ComponentState):
    model: Optional[Any]
    generator: Optional[Any]
    validation_manager: Optional[Any]
    prompt_manager: Optional[Any]
    retry_strategy: Optional[Any]
    result_formatter: Optional[Any]
    critic: Optional[Any]
    cache: Dict[str, Any]

class AdapterState(ComponentState):
    adaptee: Optional[Any]
    adaptee_cache: Dict[str, Any]
    config_cache: Dict[str, Any]
    cache: Dict[str, Any]

def create_state_manager(state_class: Type[T], **kwargs: Any) -> StateManager: ...
def create_classifier_state(**kwargs: Any) -> StateManager: ...
def create_rule_state(**kwargs: Any) -> StateManager: ...
def create_critic_state(**kwargs: Any) -> StateManager: ...
def create_model_state(**kwargs: Any) -> StateManager: ...
def create_chain_state(**kwargs: Any) -> StateManager: ...
def create_adapter_state(**kwargs: Any) -> StateManager: ...
