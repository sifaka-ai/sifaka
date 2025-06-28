"""Redis Latency Comparison: Direct vs MCP - Simplified Version

This example focuses specifically on measuring the latency differences
between direct Redis access and MCP-mediated Redis access.
"""

import asyncio
import time
import statistics
from typing import Dict, Any, List
from sifaka.storage.redis import RedisStorage
from sifaka.core.models import SifakaResult

try:
    from sifaka_tools.mcp.storage import MCPRedisStorage
except ImportError:
    print("Warning: sifaka-tools not installed. Run: pip install -e ../sifaka-tools")
    MCPRedisStorage = None


async def test_direct_redis_latency(num_operations: int = 20) -> Dict[str, Any]:
    """Test direct Redis storage latency with multiple operations."""
    print("🔴 Direct Redis Latency Test")
    print("=" * 50)
    
    # Initialize direct Redis storage
    storage = RedisStorage(
        host="localhost",
        port=6379,
        prefix="sifaka:latency:direct:",
        ttl=3600
    )
    
    print("✅ Connected directly to Redis")
    print(f"📍 Key prefix: {storage.prefix}")
    
    # Create a sample result for testing
    sample_result = SifakaResult(
        original_text="Sample text for latency testing with some content to make it realistic",
        final_text="Improved sample text for latency testing with enhanced content and better structure",
        id="latency-test-direct",
    )
    
    # Add some sample critiques
    sample_result.add_critique(
        critic="reflexion",
        feedback="Good improvement in clarity",
        suggestions=["Use simpler words", "Add examples"],
    )
    sample_result.add_critique(
        critic="self_refine",
        feedback="Structure enhanced successfully",
        suggestions=["Better transitions", "Clearer conclusion"],
    )
    
    # Test multiple operations and measure latency
    save_times = []
    load_times = []
    
    print(f"\n⏱️  Testing {num_operations} operations...")
    
    for i in range(num_operations):
        # Test SAVE operation
        save_start = time.perf_counter()
        result_id = await storage.save(sample_result)
        save_time = (time.perf_counter() - save_start) * 1000  # Convert to ms
        save_times.append(save_time)
        
        # Test LOAD operation
        load_start = time.perf_counter()
        loaded_result = await storage.load(result_id)
        load_time = (time.perf_counter() - load_start) * 1000  # Convert to ms
        load_times.append(load_time)
        
        total_time = save_time + load_time
        print(f"   Op {i+1:2d}: Save={save_time:5.1f}ms, Load={load_time:5.1f}ms | Total={total_time:6.1f}ms")
    
    # Calculate statistics
    def calc_stats(times: List[float]) -> Dict[str, float]:
        return {
            "avg": statistics.mean(times),
            "min": min(times),
            "max": max(times),
            "median": statistics.median(times),
            "p95": statistics.quantiles(times, n=20)[18] if len(times) >= 20 else max(times)  # 95th percentile
        }
    
    save_stats = calc_stats(save_times)
    load_stats = calc_stats(load_times)
    
    print(f"\n📊 Direct Redis Latency Statistics:")
    print(f"   Operation  │   Avg   │   Min   │   Max   │ Median  │   P95   ")
    print(f"   ───────────┼─────────┼─────────┼─────────┼─────────┼─────────")
    print(f"   Save       │ {save_stats['avg']:6.1f}ms│ {save_stats['min']:6.1f}ms│ {save_stats['max']:6.1f}ms│ {save_stats['median']:6.1f}ms│ {save_stats['p95']:6.1f}ms")
    print(f"   Load       │ {load_stats['avg']:6.1f}ms│ {load_stats['min']:6.1f}ms│ {load_stats['max']:6.1f}ms│ {load_stats['median']:6.1f}ms│ {load_stats['p95']:6.1f}ms")
    
    await storage.cleanup()
    
    return {
        "storage_type": "direct",
        "num_operations": num_operations,
        "save_stats": save_stats,
        "load_stats": load_stats,
        "raw_times": {
            "save": save_times,
            "load": load_times,
        }
    }


async def test_mcp_redis_latency(num_operations: int = 20) -> Dict[str, Any]:
    """Test MCP Redis storage latency with multiple operations."""
    print("\n🌐 MCP Redis Latency Test")
    print("=" * 50)
    
    if MCPRedisStorage is None:
        print("❌ MCPRedisStorage not available. Install sifaka-tools first.")
        return {}
    
    # Initialize MCP Redis storage - using stdio transport
    storage = MCPRedisStorage(
        namespace="sifaka:latency:mcp",
        timeout=10.0,
        # Stdio transport configuration
        command="uv",
        args=["--directory", "../mcp-redis", "run", "src/main.py"],
        env={
            "REDIS_HOST": "127.0.0.1",
            "REDIS_PORT": "6379",
            "REDIS_USERNAME": "default",
            "REDIS_PWD": ""
        }
    )
    
    print("✅ Connected to Redis via MCP (stdio)")
    print(f"📍 Command: {storage.command} {' '.join(storage.args)}")
    print(f"📍 Namespace: {storage.namespace}")
    
    # Create a sample result for testing
    sample_result = SifakaResult(
        original_text="Sample text for latency testing with some content to make it realistic",
        final_text="Improved sample text for latency testing with enhanced content and better structure",
        id="latency-test-mcp",
    )
    
    # Add some sample critiques
    sample_result.add_critique(
        critic="reflexion",
        feedback="Good improvement in clarity",
        suggestions=["Use simpler words", "Add examples"],
    )
    sample_result.add_critique(
        critic="self_refine", 
        feedback="Structure enhanced successfully",
        suggestions=["Better transitions", "Clearer conclusion"],
    )
    
    # Test multiple operations and measure latency
    save_times = []
    load_times = []
    
    print(f"\n⏱️  Testing {num_operations} operations...")
    
    for i in range(num_operations):
        try:
            # Test SAVE operation
            save_start = time.perf_counter()
            result_id = await storage.save(sample_result)
            save_time = (time.perf_counter() - save_start) * 1000  # Convert to ms
            save_times.append(save_time)
            
            # Test LOAD operation
            load_start = time.perf_counter()
            loaded_result = await storage.load(result_id)
            load_time = (time.perf_counter() - load_start) * 1000  # Convert to ms
            load_times.append(load_time)
            
            total_time = save_time + load_time
            print(f"   Op {i+1:2d}: Save={save_time:5.1f}ms, Load={load_time:5.1f}ms | Total={total_time:6.1f}ms")
            
        except Exception as e:
            print(f"   Op {i+1:2d}: ❌ Failed - {e}")
            break
    
    if not save_times:
        print("❌ No successful operations")
        return {"storage_type": "mcp", "error": "No successful operations"}
    
    # Calculate statistics
    def calc_stats(times: List[float]) -> Dict[str, float]:
        if not times:
            return {"avg": 0, "min": 0, "max": 0, "median": 0, "p95": 0}
        return {
            "avg": statistics.mean(times),
            "min": min(times),
            "max": max(times),
            "median": statistics.median(times),
            "p95": statistics.quantiles(times, n=20)[18] if len(times) >= 20 else max(times)  # 95th percentile
        }
    
    save_stats = calc_stats(save_times)
    load_stats = calc_stats(load_times)
    
    print(f"\n📊 MCP Redis Latency Statistics:")
    print(f"   Operation  │   Avg   │   Min   │   Max   │ Median  │   P95   ")
    print(f"   ───────────┼─────────┼─────────┼─────────┼─────────┼─────────")
    print(f"   Save       │ {save_stats['avg']:6.1f}ms│ {save_stats['min']:6.1f}ms│ {save_stats['max']:6.1f}ms│ {save_stats['median']:6.1f}ms│ {save_stats['p95']:6.1f}ms")
    print(f"   Load       │ {load_stats['avg']:6.1f}ms│ {load_stats['min']:6.1f}ms│ {load_stats['max']:6.1f}ms│ {load_stats['median']:6.1f}ms│ {load_stats['p95']:6.1f}ms")
    
    await storage.cleanup()
    
    return {
        "storage_type": "mcp",
        "num_operations": len(save_times),
        "save_stats": save_stats,
        "load_stats": load_stats,
        "raw_times": {
            "save": save_times,
            "load": load_times,
        }
    }


def compare_latencies(direct_results: Dict[str, Any], mcp_results: Dict[str, Any]):
    """Compare latencies between direct and MCP Redis."""
    print("\n\n🏁 Latency Comparison Summary")
    print("=" * 70)
    
    if not direct_results or not mcp_results or "error" in mcp_results:
        print("❌ Cannot compare - one or both tests failed")
        return
    
    operations = ["save", "load"]
    
    print(f"\n📈 Average Latency Comparison:")
    print(f"   Operation  │  Direct  │   MCP    │ Overhead │ Slowdown ")
    print(f"   ───────────┼──────────┼──────────┼──────────┼──────────")
    
    total_overhead = 0
    for op in operations:
        direct_avg = direct_results[f"{op}_stats"]["avg"]
        mcp_avg = mcp_results[f"{op}_stats"]["avg"]
        overhead = mcp_avg - direct_avg
        slowdown = (mcp_avg / direct_avg) if direct_avg > 0 else 0
        total_overhead += overhead
        
        print(f"   {op.capitalize():<10} │ {direct_avg:7.1f}ms│ {mcp_avg:7.1f}ms│ +{overhead:6.1f}ms│ {slowdown:6.1f}x")
    
    print(f"   ───────────┼──────────┼──────────┼──────────┼──────────")
    print(f"   {'Total':<10} │          │          │ +{total_overhead:6.1f}ms│")
    
    # P95 comparison
    print(f"\n📊 95th Percentile Latency Comparison:")
    print(f"   Operation  │  Direct  │   MCP    │ Overhead │ Slowdown ")
    print(f"   ───────────┼──────────┼──────────┼──────────┼──────────")
    
    for op in operations:
        direct_p95 = direct_results[f"{op}_stats"]["p95"]
        mcp_p95 = mcp_results[f"{op}_stats"]["p95"]
        overhead = mcp_p95 - direct_p95
        slowdown = (mcp_p95 / direct_p95) if direct_p95 > 0 else 0
        
        print(f"   {op.capitalize():<10} │ {direct_p95:7.1f}ms│ {mcp_p95:7.1f}ms│ +{overhead:6.1f}ms│ {slowdown:6.1f}x")
    
    # Key insights
    print(f"\n💡 Key Insights:")
    avg_slowdown = sum((mcp_results[f"{op}_stats"]["avg"] / direct_results[f"{op}_stats"]["avg"]) 
                      for op in operations) / len(operations)
    print(f"   • MCP Redis is {avg_slowdown:.1f}x slower on average")
    print(f"   • Total overhead per operation cycle: +{total_overhead:.1f}ms")
    print(f"   • MCP overhead comes from process communication and JSON-RPC protocol")
    print(f"   • Direct Redis has sub-millisecond latency for most operations")
    print(f"   • MCP Redis suitable for non-latency-critical applications")


async def main():
    """Run the latency comparison."""
    print("⚡ Redis Latency Comparison: Direct vs MCP")
    print("=" * 70)
    print()
    
    # Check prerequisites
    print("📋 Prerequisites:")
    print("1. Start Redis with Docker: docker run -d -p 6379:6379 redis")
    print("2. Install sifaka-tools: pip install -e ../sifaka-tools")
    print()
    
    num_ops = 20
    print(f"🎯 Running {num_ops} operations for each storage type...")
    print()
    
    # Run direct Redis test
    direct_results = await test_direct_redis_latency(num_ops)
    
    # Run MCP Redis test
    mcp_results = await test_mcp_redis_latency(num_ops)
    
    # Compare results
    compare_latencies(direct_results, mcp_results)


if __name__ == "__main__":
    asyncio.run(main())
