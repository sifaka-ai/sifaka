# MCP server forces confusing field-arrays format instead of supporting standard list-of-dicts format

## Problem Description

The MCP server currently forces a confusing "field arrays" data format that doesn't align with standard pymilvus usage patterns and creates unnecessary complexity for single-record insertions.

## Current Behavior

The MCP server validation requires this format:
```python
# What MCP server forces:
data = {
    "vector": [1.0, 2.0, 3.0],     # Direct vector
    "key": ["test_key"],           # Field as single-element array  
    "content": ["test_data"]       # Field as single-element array
}
```

This results in confusing storage where fields are arrays:
```python
# Query result:
{'key': ['test_key'], 'content': ['test_data'], 'id': 123}
```

And requires awkward array indexing in queries:
```python
# Query filter:
filter_expr = "key[0] == 'test_key'"  # Why [0]?
```

## Expected Behavior

The MCP server should support the standard pymilvus list-of-dicts format:
```python
# What should be supported:
data = [{"vector": [1.0, 2.0, 3.0], "key": "test_key", "content": "test_data"}]
```

This would result in clean storage:
```python
# Query result:
{'key': 'test_key', 'content': 'test_data', 'id': 123}
```

And natural queries:
```python
# Query filter:
filter_expr = "key == 'test_key'"  # Much cleaner!
```

## Root Cause

The issue is in the tool definition validation schema:

**File:** `src/mcp_server_milvus/server.py`
**Line:** ~710

```python
async def milvus_insert_data(
    collection_name: str, data: dict[str, list[Any]], ctx: Context = None
) -> str:
```

The `data: dict[str, list[Any]]` type annotation forces the field-arrays format, but pymilvus actually supports both:
1. **List of dicts** (preferred): `[{"field1": "value1", "field2": "value2"}]`
2. **Field arrays** (current): `{"field1": ["value1"], "field2": ["value2"]}`

## Proposed Solution

Update the type annotation to accept list-of-dicts:

```python
async def milvus_insert_data(
    collection_name: str, data: list[dict[str, Any]], ctx: Context = None
) -> str:
```

And update the corresponding method signature:
```python
async def insert_data(
    self, collection_name: str, data: list[dict[str, Any]]
) -> dict[str, Any]:
```

## Benefits

1. **Cleaner data model**: Fields stored as direct values, not arrays
2. **Natural queries**: `key == 'value'` instead of `key[0] == 'value'`
3. **Standard pymilvus patterns**: Aligns with official pymilvus documentation
4. **Better UX**: Less confusing for developers using the MCP server
5. **Backward compatibility**: pymilvus supports both formats

## Impact

This is a breaking change for existing MCP server users, but it aligns with standard pymilvus usage and significantly improves the developer experience.

## Workaround

Currently, developers must:
1. Use the confusing field-arrays format
2. Use array indexing in queries (`key[0] == 'value'`)
3. Handle array extraction when parsing results

## Environment

- MCP Server Version: Latest
- Pymilvus Version: Latest
- Python Version: 3.11+
