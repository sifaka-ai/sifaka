# Configuration Errors and Solutions

This guide helps you resolve configuration-related issues when setting up and using Sifaka.

## Configuration Validation Script

Run this script to validate your Sifaka configuration:

```python
#!/usr/bin/env python3
"""Validate Sifaka configuration."""

import os
from typing import Dict, Any

def validate_configuration():
    """Comprehensive configuration validation."""
    print("🔧 Sifaka Configuration Validation")
    print("=" * 50)

    issues = []
    warnings = []

    # Check environment variables
    env_vars = {
        "OPENAI_API_KEY": "OpenAI API access",
        "ANTHROPIC_API_KEY": "Anthropic API access",
        "HUGGINGFACE_API_KEY": "HuggingFace API access (optional)",
        "REDIS_URL": "Redis connection (optional)",
        "MILVUS_HOST": "Milvus connection (optional)"
    }

    print("\n🔑 Environment Variables:")
    for var, description in env_vars.items():
        value = os.getenv(var)
        if value:
            # Mask sensitive values
            masked = f"{value[:8]}..." if len(value) > 8 else "***"
            print(f"✅ {var}: {masked} ({description})")
        else:
            status = "⚠️" if "optional" in description else "❌"
            print(f"{status} {var}: Not set ({description})")
            if "optional" not in description:
                issues.append(f"Missing required environment variable: {var}")

    # Test model configurations
    print("\n🤖 Model Configuration:")
    test_model_configs(issues, warnings)

    # Test storage configurations
    print("\n💾 Storage Configuration:")
    test_storage_configs(issues, warnings)

    # Summary
    print("\n" + "=" * 50)
    if issues:
        print("❌ Configuration Issues Found:")
        for issue in issues:
            print(f"  • {issue}")

    if warnings:
        print("⚠️  Configuration Warnings:")
        for warning in warnings:
            print(f"  • {warning}")

    if not issues and not warnings:
        print("✅ Configuration looks good!")

    return len(issues) == 0

def test_model_configs(issues, warnings):
    """Test model configurations."""
    from sifaka.models import create_model

    # Test OpenAI
    try:
        if os.getenv("OPENAI_API_KEY"):
            model = create_model("openai:gpt-3.5-turbo")
            print("✅ OpenAI: Configuration valid")
        else:
            print("⚠️  OpenAI: API key not set")
            warnings.append("OpenAI API key not configured")
    except Exception as e:
        print(f"❌ OpenAI: {e}")
        issues.append(f"OpenAI configuration error: {e}")

    # Test Anthropic
    try:
        if os.getenv("ANTHROPIC_API_KEY"):
            model = create_model("anthropic:claude-3-haiku")
            print("✅ Anthropic: Configuration valid")
        else:
            print("⚠️  Anthropic: API key not set")
            warnings.append("Anthropic API key not configured")
    except Exception as e:
        print(f"❌ Anthropic: {e}")
        issues.append(f"Anthropic configuration error: {e}")

    # Test Mock (should always work)
    try:
        model = create_model("mock:test-model")
        print("✅ Mock: Available for testing")
    except Exception as e:
        print(f"❌ Mock: {e}")
        issues.append(f"Mock model error: {e}")

def test_storage_configs(issues, warnings):
    """Test storage configurations."""
    # Test Memory storage
    try:
        from sifaka.storage import MemoryStorage
        storage = MemoryStorage()
        print("✅ Memory: Available")
    except Exception as e:
        print(f"❌ Memory: {e}")
        issues.append(f"Memory storage error: {e}")

    # Test Redis if configured
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        try:
            import redis
            r = redis.from_url(redis_url)
            r.ping()
            print("✅ Redis: Connection successful")
        except ImportError:
            print("❌ Redis: redis package not installed")
            issues.append("Redis package not installed: pip install redis")
        except Exception as e:
            print(f"❌ Redis: {e}")
            issues.append(f"Redis connection error: {e}")
    else:
        print("⚠️  Redis: Not configured")

    # Test Milvus if configured
    milvus_host = os.getenv("MILVUS_HOST")
    if milvus_host:
        try:
            from pymilvus import connections
            connections.connect("test", host=milvus_host, port=os.getenv("MILVUS_PORT", "19530"))
            print("✅ Milvus: Connection successful")
        except ImportError:
            print("❌ Milvus: pymilvus package not installed")
            issues.append("Milvus package not installed: pip install pymilvus")
        except Exception as e:
            print(f"❌ Milvus: {e}")
            issues.append(f"Milvus connection error: {e}")
    else:
        print("⚠️  Milvus: Not configured")

if __name__ == "__main__":
    validate_configuration()
```

## Common Configuration Errors

### 1. API Key Configuration Issues

#### Problem: Invalid or missing API keys

**Symptoms:**
- `AuthenticationError` or `PermissionError`
- "Invalid API key" messages
- 401 Unauthorized responses

**Solutions:**

```bash
# 1. Check if API keys are set
env | grep -E "(OPENAI|ANTHROPIC)_API_KEY"

# 2. Set API keys in environment
export OPENAI_API_KEY="sk-your-openai-key-here"
export ANTHROPIC_API_KEY="sk-ant-your-anthropic-key-here"

# 3. Use .env file (recommended)
# Create .env file in your project root:
echo "OPENAI_API_KEY=sk-your-openai-key-here" > .env
echo "ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here" >> .env

# 4. Load .env file in Python
pip install python-dotenv
```

```python
# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Verify API keys are loaded
import os
print("OpenAI API Key:", "✅ Set" if os.getenv("OPENAI_API_KEY") else "❌ Missing")
print("Anthropic API Key:", "✅ Set" if os.getenv("ANTHROPIC_API_KEY") else "❌ Missing")
```

#### Problem: API key format errors

**Symptoms:**
- "Invalid API key format" errors
- Authentication fails despite key being set

**Solutions:**

```python
def validate_api_key_format():
    """Validate API key formats."""
    import os

    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    # OpenAI key format: sk-...
    if openai_key:
        if openai_key.startswith("sk-") and len(openai_key) > 20:
            print("✅ OpenAI API key format looks correct")
        else:
            print("❌ OpenAI API key format incorrect (should start with 'sk-')")

    # Anthropic key format: sk-ant-...
    if anthropic_key:
        if anthropic_key.startswith("sk-ant-") and len(anthropic_key) > 30:
            print("✅ Anthropic API key format looks correct")
        else:
            print("❌ Anthropic API key format incorrect (should start with 'sk-ant-')")

validate_api_key_format()
```

### 2. Model Configuration Errors

#### Problem: Unsupported model names

**Symptoms:**
- "Model not found" errors
- "Unsupported model" messages

**Solutions:**

```python
# 1. Use correct model specifications
from sifaka.models import create_model

# ✅ Correct model specifications
correct_models = [
    "openai:gpt-4",
    "openai:gpt-3.5-turbo",
    "anthropic:claude-3-opus",
    "anthropic:claude-3-sonnet",
    "anthropic:claude-3-haiku",
    "gemini:gemini-1.5-flash",
    "gemini:gemini-1.5-pro",
    "mock:test-model"
]

# ❌ Common incorrect specifications
incorrect_models = [
    "gpt-4",  # Missing provider prefix
    "openai:gpt4",  # Wrong model name
    "claude-3",  # Missing provider prefix
    "anthropic:claude3"  # Wrong model name
]

# Test model creation
for model_spec in correct_models:
    try:
        model = create_model(model_spec)
        print(f"✅ {model_spec}: Valid")
    except Exception as e:
        print(f"❌ {model_spec}: {e}")
```

#### Problem: Model parameter errors

**Symptoms:**
- Parameter validation errors
- Unexpected model behavior

**Solutions:**

```python
# 1. Use valid parameter ranges
model = create_model(
    "openai:gpt-4",
    temperature=0.7,  # 0.0 to 2.0
    max_tokens=1000,  # Positive integer
    timeout=30        # Seconds
)

# 2. Validate parameters before use
def validate_model_params(**params):
    """Validate model parameters."""
    issues = []

    if "temperature" in params:
        temp = params["temperature"]
        if not 0.0 <= temp <= 2.0:
            issues.append(f"Temperature {temp} not in range [0.0, 2.0]")

    if "max_tokens" in params:
        tokens = params["max_tokens"]
        if not isinstance(tokens, int) or tokens <= 0:
            issues.append(f"max_tokens {tokens} must be positive integer")

    if "timeout" in params:
        timeout = params["timeout"]
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            issues.append(f"timeout {timeout} must be positive number")

    return issues

# Usage
params = {"temperature": 0.7, "max_tokens": 1000, "timeout": 30}
issues = validate_model_params(**params)
if issues:
    print("Parameter issues:", issues)
else:
    model = create_model("openai:gpt-4", **params)
```

### 3. Storage Configuration Errors

#### Problem: Redis connection issues

**Symptoms:**
- `ConnectionError` to Redis
- "Connection refused" errors

**Solutions:**

```python
# 1. Test Redis connection
def test_redis_connection():
    """Test Redis connection with different configurations."""
    import redis

    # Common Redis configurations
    configs = [
        {"host": "localhost", "port": 6379},
        {"host": "127.0.0.1", "port": 6379},
        {"url": "redis://localhost:6379"},
        {"url": os.getenv("REDIS_URL")} if os.getenv("REDIS_URL") else None
    ]

    for config in configs:
        if config is None:
            continue

        try:
            if "url" in config:
                r = redis.from_url(config["url"])
            else:
                r = redis.Redis(**config)

            r.ping()
            print(f"✅ Redis connection successful: {config}")
            return config
        except Exception as e:
            print(f"❌ Redis connection failed: {config} - {e}")

    print("❌ All Redis connections failed")
    return None

# 2. Start Redis if not running
def start_redis_docker():
    """Start Redis using Docker."""
    import subprocess

    try:
        # Check if Redis is already running
        result = subprocess.run(["redis-cli", "ping"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Redis is already running")
            return True
    except FileNotFoundError:
        pass

    try:
        # Start Redis with Docker
        subprocess.run([
            "docker", "run", "-d",
            "--name", "redis-sifaka",
            "-p", "6379:6379",
            "redis:alpine"
        ], check=True)
        print("✅ Redis started with Docker")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start Redis: {e}")
        return False

test_redis_connection()
```

#### Problem: Milvus configuration issues

**Symptoms:**
- Milvus connection errors
- Collection creation failures

**Solutions:**

```python
# 1. Test Milvus connection
def test_milvus_connection():
    """Test Milvus connection and setup."""
    try:
        from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType

        # Connect to Milvus
        host = os.getenv("MILVUS_HOST", "localhost")
        port = os.getenv("MILVUS_PORT", "19530")

        connections.connect("default", host=host, port=port)
        print(f"✅ Connected to Milvus at {host}:{port}")

        # Test basic operations
        collections = utility.list_collections()
        print(f"✅ Available collections: {collections}")

        return True

    except ImportError:
        print("❌ pymilvus not installed: pip install pymilvus")
        return False
    except Exception as e:
        print(f"❌ Milvus connection failed: {e}")
        print("Solutions:")
        print("  1. Start Milvus: bash standalone_embed.sh start")
        print("  2. Check MILVUS_HOST and MILVUS_PORT environment variables")
        print("  3. Verify Milvus is running: curl http://localhost:9091/health")
        return False

# 2. Create test collection
def create_test_collection():
    """Create a test collection to verify Milvus setup."""
    try:
        from pymilvus import Collection, FieldSchema, CollectionSchema, DataType

        # Define collection schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768)
        ]

        schema = CollectionSchema(fields, "Test collection for Sifaka")
        collection = Collection("sifaka_test", schema)

        print("✅ Test collection created successfully")

        # Clean up
        collection.drop()
        print("✅ Test collection cleaned up")

        return True

    except Exception as e:
        print(f"❌ Collection creation failed: {e}")
        return False

test_milvus_connection()
```

### 4. Chain Configuration Errors

#### Problem: Invalid chain parameters

**Symptoms:**
- Chain creation fails
- Unexpected chain behavior

**Solutions:**

```python
# 1. Validate chain configuration
def validate_chain_config(**config):
    """Validate chain configuration parameters."""
    issues = []

    # Check required parameters
    if "model" not in config:
        issues.append("Model is required")

    if "prompt" not in config:
        issues.append("Prompt is required")

    # Check optional parameters
    if "max_improvement_iterations" in config:
        iterations = config["max_improvement_iterations"]
        if not isinstance(iterations, int) or iterations < 0:
            issues.append("max_improvement_iterations must be non-negative integer")

    if "apply_improvers_on_validation_failure" in config:
        apply_improvers = config["apply_improvers_on_validation_failure"]
        if not isinstance(apply_improvers, bool):
            issues.append("apply_improvers_on_validation_failure must be boolean")

    return issues

# 2. Create chain with validation
from sifaka.agents import create_pydantic_chain
from sifaka.models import create_model
from pydantic_ai import Agent

def create_validated_chain(agent, **config):
    """Create PydanticAI chain with configuration validation."""
    issues = validate_chain_config(**config)

    if issues:
        print("❌ Chain configuration issues:")
        for issue in issues:
            print(f"  • {issue}")
        return None

    try:
        chain = create_pydantic_chain(agent=agent, **config)
        print("✅ PydanticAI Chain created successfully")
        return chain
    except Exception as e:
        print(f"❌ Chain creation failed: {e}")
        return None

# Usage
model = create_model("mock:test-model")
chain = create_validated_chain(
    model=model,
    prompt="Test prompt",
    max_improvement_iterations=3
)
```

### 5. Validator Configuration Errors

#### Problem: Invalid validator parameters

**Solutions:**

```python
# 1. Validate validator configurations
from sifaka.validators import LengthValidator, RegexValidator

def create_validated_validators():
    """Create validators with proper configuration."""
    validators = []

    # Length validator with validation
    try:
        length_validator = LengthValidator(
            min_length=10,
            max_length=1000,
            min_words=5,
            max_words=200
        )
        validators.append(length_validator)
        print("✅ Length validator created")
    except Exception as e:
        print(f"❌ Length validator failed: {e}")

    # Regex validator with validation
    try:
        regex_validator = RegexValidator(
            required_patterns=[r'\b\w+@\w+\.\w+\b'],  # Email pattern
            forbidden_patterns=[r'\b(spam|scam)\b'],
            name="EmailValidator"
        )
        validators.append(regex_validator)
        print("✅ Regex validator created")
    except Exception as e:
        print(f"❌ Regex validator failed: {e}")

    return validators

validators = create_validated_validators()
```

## Environment-Specific Configuration

### Development Configuration

```python
# development.py
import os
from sifaka.agents import create_pydantic_chain
from sifaka.models import create_model
from sifaka.storage import MemoryStorage
from pydantic_ai import Agent

def create_dev_config():
    """Create development configuration."""
    agent = Agent("mock:test-model", system_prompt="You are a helpful assistant.")
    return create_pydantic_chain(
        agent=agent,
        storage=MemoryStorage(),  # No external dependencies
        validators=[],
        critics=[],
        max_improvement_iterations=1,  # Fast iterations
    )

# Usage
chain = create_dev_config()  # Returns a ready-to-use PydanticAI chain
```

### Production Configuration

```python
# production.py
import os
from sifaka import Chain
from sifaka.models import create_model
from sifaka.storage import RedisStorage, CachedStorage, MemoryStorage

def create_prod_config():
    """Create production configuration."""
    # Validate required environment variables
    required_vars = ["OPENAI_API_KEY", "REDIS_URL"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        raise ValueError(f"Missing required environment variables: {missing_vars}")

    # Create storage with caching
    storage = CachedStorage(
        cache=MemoryStorage(max_size=1000),
        persistence=RedisStorage(redis_url=os.getenv("REDIS_URL"))
    )

    return {
        "model": create_model("openai:gpt-4", timeout=60),
        "storage": storage,
        "max_improvement_iterations": 3,
        "debug": False
    }

# Usage
try:
    config = create_prod_config()
    chain = Chain(**config)
    print("✅ Production configuration ready")
except Exception as e:
    print(f"❌ Production configuration failed: {e}")
```

## Configuration Best Practices

### 1. Use Configuration Files

```python
# config.py
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class SifakaConfig:
    """Sifaka configuration class."""
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    redis_url: Optional[str] = None
    milvus_host: Optional[str] = None
    milvus_port: int = 19530
    debug: bool = False

    def __post_init__(self):
        """Load from environment if not provided."""
        self.openai_api_key = self.openai_api_key or os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = self.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        self.redis_url = self.redis_url or os.getenv("REDIS_URL")
        self.milvus_host = self.milvus_host or os.getenv("MILVUS_HOST", "localhost")

    def validate(self):
        """Validate configuration."""
        issues = []

        if not self.openai_api_key and not self.anthropic_api_key:
            issues.append("At least one API key (OpenAI or Anthropic) is required")

        if self.openai_api_key and not self.openai_api_key.startswith("sk-"):
            issues.append("OpenAI API key should start with 'sk-'")

        if self.anthropic_api_key and not self.anthropic_api_key.startswith("sk-ant-"):
            issues.append("Anthropic API key should start with 'sk-ant-'")

        return issues

# Usage
config = SifakaConfig()
issues = config.validate()
if issues:
    print("Configuration issues:", issues)
else:
    print("Configuration valid")
```

### 2. Environment-Specific Configs

```bash
# .env.development
OPENAI_API_KEY=sk-dev-key
DEBUG=true
LOG_LEVEL=DEBUG

# .env.production
OPENAI_API_KEY=sk-prod-key
REDIS_URL=redis://prod-redis:6379
DEBUG=false
LOG_LEVEL=INFO
```

### 3. Configuration Validation

```python
def validate_full_configuration():
    """Comprehensive configuration validation."""
    print("🔧 Full Configuration Validation")
    print("=" * 50)

    # Run the validation script from the beginning
    return validate_configuration()

# Run validation before starting application
if __name__ == "__main__":
    if validate_full_configuration():
        print("✅ Ready to start Sifaka application")
    else:
        print("❌ Fix configuration issues before proceeding")
        exit(1)
```

Most configuration errors can be prevented by:
1. **Using environment variables** for sensitive data
2. **Validating configuration** before use
3. **Testing connections** during startup
4. **Using configuration classes** for type safety
5. **Following naming conventions** for models and parameters
