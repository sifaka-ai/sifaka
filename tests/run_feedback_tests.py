#!/usr/bin/env python3
"""
Test runner for comprehensive feedback integration tests.

This script runs all the feedback integration tests and provides a summary
of which critics and validation scenarios are working correctly.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add the parent directory to the path so we can import sifaka
sys.path.insert(0, str(Path(__file__).parent.parent))

from sifaka import Sifaka


async def test_basic_validation_feedback():
    """Test basic validation feedback integration."""
    print("🔍 Testing basic validation feedback...")
    try:
        result = await (
            Sifaka('Write a very long explanation about AI')
            .min_length(50)
            .max_length(100)  # Force failure
            .max_iterations(2)
            .improve()
        )
        
        # Check validation feedback in prompt
        if len(result.generations) > 1:
            prompt = result.generations[1].user_prompt
            has_validation = 'Validation Results' in prompt
            has_failed = 'FAILED' in prompt
            print(f"  ✅ Validation feedback in prompt: {has_validation}")
            print(f"  ✅ Validation failures shown: {has_failed}")
            return has_validation and has_failed
        return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


async def test_reflexion_critic():
    """Test ReflexionCritic feedback."""
    print("🧠 Testing ReflexionCritic...")
    try:
        result = await (
            Sifaka('AI is good')
            .max_length(2000)
            .max_iterations(2)
            .with_reflexion()
            .improve()
        )
        
        reflexion_critiques = [c for c in result.critiques if c.critic == 'ReflexionCritic']
        has_critiques = len(reflexion_critiques) > 0
        
        # Check feedback in prompt
        has_in_prompt = False
        if len(result.generations) > 1:
            prompt = result.generations[1].user_prompt
            has_in_prompt = 'ReflexionCritic' in prompt
        
        print(f"  ✅ ReflexionCritic provided feedback: {has_critiques}")
        print(f"  ✅ ReflexionCritic in prompt: {has_in_prompt}")
        return has_critiques and has_in_prompt
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


async def test_constitutional_critic():
    """Test ConstitutionalCritic feedback."""
    print("⚖️ Testing ConstitutionalCritic...")
    try:
        result = await (
            Sifaka('Write about AI ethics')
            .max_length(2000)
            .max_iterations(2)
            .with_constitutional()
            .improve()
        )
        
        constitutional_critiques = [c for c in result.critiques if c.critic == 'ConstitutionalCritic']
        has_critiques = len(constitutional_critiques) > 0
        
        # Check feedback in prompt
        has_in_prompt = False
        if len(result.generations) > 1:
            prompt = result.generations[1].user_prompt
            has_in_prompt = 'ConstitutionalCritic' in prompt
        
        print(f"  ✅ ConstitutionalCritic provided feedback: {has_critiques}")
        print(f"  ✅ ConstitutionalCritic in prompt: {has_in_prompt}")
        return has_critiques and has_in_prompt
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


async def test_self_consistency_critic():
    """Test SelfConsistencyCritic feedback."""
    print("🔄 Testing SelfConsistencyCritic...")
    try:
        result = await (
            Sifaka('Write about neural networks')
            .max_length(2000)
            .max_iterations(2)
            .with_self_consistency()
            .improve()
        )
        
        consistency_critiques = [c for c in result.critiques if c.critic == 'SelfConsistencyCritic']
        has_critiques = len(consistency_critiques) > 0
        
        # Check feedback in prompt
        has_in_prompt = False
        if len(result.generations) > 1:
            prompt = result.generations[1].user_prompt
            has_in_prompt = 'SelfConsistencyCritic' in prompt
        
        print(f"  ✅ SelfConsistencyCritic provided feedback: {has_critiques}")
        print(f"  ✅ SelfConsistencyCritic in prompt: {has_in_prompt}")
        return has_critiques and has_in_prompt
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


async def test_multiple_critics():
    """Test multiple critics working together."""
    print("👥 Testing multiple critics together...")
    try:
        result = await (
            Sifaka('Write about artificial intelligence')
            .max_length(2000)
            .max_iterations(2)
            .with_reflexion()
            .with_constitutional()
            .improve()
        )
        
        critic_types = {c.critic for c in result.critiques}
        has_reflexion = 'ReflexionCritic' in critic_types
        has_constitutional = 'ConstitutionalCritic' in critic_types
        
        # Check both in prompt
        both_in_prompt = False
        if len(result.generations) > 1:
            prompt = result.generations[1].user_prompt
            both_in_prompt = 'ReflexionCritic' in prompt and 'ConstitutionalCritic' in prompt
        
        print(f"  ✅ Both critics provided feedback: {has_reflexion and has_constitutional}")
        print(f"  ✅ Both critics in prompt: {both_in_prompt}")
        return has_reflexion and has_constitutional and both_in_prompt
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


async def test_feedback_ordering():
    """Test that feedback appears in correct order."""
    print("📋 Testing feedback ordering...")
    try:
        result = await (
            Sifaka('Write a comprehensive guide')
            .min_length(100)
            .max_length(500)  # Force validation failure
            .max_iterations(2)
            .with_reflexion()
            .improve()
        )
        
        if len(result.generations) > 1:
            prompt = result.generations[1].user_prompt
            
            # Find positions
            validation_pos = prompt.find('Validation Results')
            critic_pos = prompt.find('Critic')
            text_pos = prompt.find('Previous attempt:')
            
            correct_order = validation_pos < critic_pos < text_pos
            feedback_first = prompt.find('Improve the following text based on this feedback') < text_pos
            
            print(f"  ✅ Validation → Critics → Text order: {correct_order}")
            print(f"  ✅ Feedback before text: {feedback_first}")
            return correct_order and feedback_first
        return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


async def test_weight_display():
    """Test that weights are displayed correctly."""
    print("⚖️ Testing weight display...")
    try:
        result = await (
            Sifaka('Write about AI')
            .validation_weight(0.7)  # 70% validation, 30% critics
            .max_length(500)
            .max_iterations(2)
            .with_reflexion()
            .improve()
        )
        
        if len(result.generations) > 1:
            prompt = result.generations[1].user_prompt
            has_70_percent = '70%' in prompt
            has_30_percent = '30%' in prompt
            
            print(f"  ✅ 70% validation weight shown: {has_70_percent}")
            print(f"  ✅ 30% critic weight shown: {has_30_percent}")
            return has_70_percent and has_30_percent
        return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


async def main():
    """Run all feedback integration tests."""
    print("🚀 Starting Comprehensive Feedback Integration Tests")
    print("=" * 60)
    
    start_time = time.time()
    
    tests = [
        ("Basic Validation Feedback", test_basic_validation_feedback),
        ("ReflexionCritic", test_reflexion_critic),
        ("ConstitutionalCritic", test_constitutional_critic),
        ("SelfConsistencyCritic", test_self_consistency_critic),
        ("Multiple Critics", test_multiple_critics),
        ("Feedback Ordering", test_feedback_ordering),
        ("Weight Display", test_weight_display),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📝 Running: {test_name}")
        try:
            success = await test_func()
            results.append((test_name, success))
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"   {status}")
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    elapsed = time.time() - start_time
    print(f"⏱️ Total time: {elapsed:.1f} seconds")
    
    if passed == total:
        print("\n🎉 All feedback integration tests PASSED!")
        return 0
    else:
        print(f"\n⚠️ {total - passed} tests FAILED - feedback integration needs attention")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
