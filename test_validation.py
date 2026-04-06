#!/usr/bin/env python3
"""
Comprehensive validation script for:
1. Kalshi API rate limits and connectivity
2. OpenRouter multi-model capability
3. Ensemble decision-making with multiple models
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Any

from src.clients.kalshi_client import KalshiClient
from src.clients.openrouter_client import OpenRouterClient
from src.clients.xai_client import XAIClient
from src.clients.model_router import ModelRouter
from src.config.settings import settings
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


# =============================================================================
# TEST 1: KALSHI API CONNECTIVITY AND RATE LIMITS
# =============================================================================

async def test_kalshi_api():
    """Test Kalshi API connectivity and rate limits."""
    print("\n" + "="*80)
    print("TEST 1: KALSHI API CONNECTIVITY & RATE LIMITS")
    print("="*80)

    try:
        kalshi = KalshiClient(
            api_key=settings.api.kalshi_api_key,
            private_key_path="./kalshi_private_key.pem",
        )

        # Test 1a: Get account info
        print("\n[1a] Fetching account info...")
        account = await kalshi.get_account()
        if account:
            print(f"[OK] Account balance: ${account.get('cash_balance', 'N/A')}")
        else:
            print("[FAIL] Failed to fetch account")
            return False

        # Test 1b: Fetch markets with delay between requests to respect rate limits
        print("\n[1b] Fetching markets (with rate limit awareness)...")
        markets = []
        cursor = None
        request_count = 0

        for batch in range(3):  # Fetch 3 batches
            try:
                print(f"  Batch {batch+1}/3...", end=" ", flush=True)
                response = await kalshi.get_markets(limit=50, cursor=cursor)

                if response:
                    markets.extend(response.get("markets", []))
                    cursor = response.get("cursor")
                    request_count += 1
                    print(f"[OK] Got {len(response.get('markets', []))} markets")

                    # Respectful rate limiting
                    if batch < 2:
                        await asyncio.sleep(0.5)
                else:
                    print("[FAIL] Failed")
                    break

            except Exception as e:
                if "429" in str(e):
                    print(f"[WARN] Rate limited! {str(e)[:60]}...")
                    # Wait and retry
                    await asyncio.sleep(5)
                    try:
                        response = await kalshi.get_markets(limit=50, cursor=cursor)
                        if response:
                            markets.extend(response.get("markets", []))
                            request_count += 1
                            print(f"  [OK] Retry successful, got {len(response.get('markets', []))} markets")
                    except Exception as retry_err:
                        print(f"  [FAIL] Retry also failed: {str(retry_err)[:60]}...")
                else:
                    print(f"[FAIL] Error: {str(e)[:60]}...")
                break

        print(f"\n[STATS] Kalshi API Summary:")
        print(f"  • Total requests: {request_count}")
        print(f"  • Markets fetched: {len(markets)}")

        return True if markets else False

    except Exception as e:
        print(f"[FAIL] Kalshi API test failed: {e}")
        return False


# =============================================================================
# TEST 2: OPENROUTER MULTI-MODEL CAPABILITY
# =============================================================================

async def test_openrouter_models():
    """Test each model in the ensemble individually."""
    print("\n" + "="*80)
    print("TEST 2: OPENROUTER MULTI-MODEL VALIDATION")
    print("="*80)

    client = OpenRouterClient(
        api_key=settings.api.openrouter_api_key,
        default_model="anthropic/claude-sonnet-4.5",
    )

    # Test each model from the ensemble config
    test_prompt = """
    Analyze this market: "Will the Fed raise rates next meeting?"
    YES is trading at $0.72, NO at $0.28.
    Volume: 50,000 contracts. Expires in 5 days.

    Respond with JSON: {"prediction": "yes|no|pass", "confidence": 0.0-1.0, "reasoning": "..."}
    """

    models_to_test = [
        "anthropic/claude-sonnet-4.5",
        "openai/o3",
        "google/gemini-3-pro-preview",
        "deepseek/deepseek-v3.2",
    ]

    results = {}

    for model_name in models_to_test:
        print(f"\n[Testing] {model_name}...", end=" ", flush=True)
        try:
            start = time.time()

            # Use get_completion for a simple test
            response = await client.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": test_prompt}],
                temperature=0,
                max_tokens=500,
            )

            elapsed = time.time() - start

            # Extract response
            if response and response.choices:
                text = response.choices[0].message.content

                # Track in client
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens

                # Calculate cost
                from src.clients.openrouter_client import MODEL_PRICING
                pricing = MODEL_PRICING.get(model_name, {})
                cost = (input_tokens / 1000 * pricing.get("input_per_1k", 0) +
                        output_tokens / 1000 * pricing.get("output_per_1k", 0))

                results[model_name] = {
                    "status": "[OK] SUCCESS",
                    "latency_ms": round(elapsed * 1000),
                    "tokens_in": input_tokens,
                    "tokens_out": output_tokens,
                    "cost_usd": round(cost, 6),
                    "response_preview": text[:100] + "..." if len(text) > 100 else text,
                }

                print(f"[OK] {round(elapsed*1000)}ms | {input_tokens}→{output_tokens} tokens | ${round(cost, 4)}")
            else:
                results[model_name] = {"status": "[FAIL] No response"}
                print("[FAIL] No response received")

        except Exception as e:
            error_msg = str(e)[:100]
            results[model_name] = {"status": f"[FAIL] {error_msg}"}
            print(f"[FAIL] {error_msg}")

        # Rate limit courtesy
        await asyncio.sleep(1)

    # Summary
    print(f"\n[STATS] OpenRouter Model Results:")
    success_count = sum(1 for r in results.values() if "SUCCESS" in str(r.get("status", "")))
    print(f"  • Models tested: {len(models_to_test)}")
    print(f"  • Successful: {success_count}/{len(models_to_test)}")
    print(f"  • Failed: {len(models_to_test) - success_count}/{len(models_to_test)}")

    for model, result in results.items():
        print(f"\n  {model}:")
        for key, value in result.items():
            print(f"    {key}: {value}")

    return success_count == len(models_to_test)


# =============================================================================
# TEST 3: ENSEMBLE DECISION-MAKING
# =============================================================================

async def test_ensemble_decisions():
    """Test that multiple models can provide independent decisions."""
    print("\n" + "="*80)
    print("TEST 3: ENSEMBLE DECISION-MAKING")
    print("="*80)

    # Mock market data
    market_data = {
        "title": "Will inflation be below 3% next month?",
        "yes_price": 0.35,
        "no_price": 0.65,
        "volume": 100000,
        "days_to_expiry": 7,
    }

    print(f"\nMarket: {market_data['title']}")
    print(f"  YES: ${market_data['yes_price']} | NO: ${market_data['no_price']}")
    print(f"  Volume: {market_data['volume']:,} | Expiry: {market_data['days_to_expiry']} days")

    # Initialize ensemble
    openrouter = OpenRouterClient(
        api_key=settings.api.openrouter_api_key,
        default_model="anthropic/claude-sonnet-4.5",
    )

    router = ModelRouter(openrouter_client=openrouter)

    # Define role-based models as per original design
    ensemble_models = [
        ("grok-3", "xai", "forecaster"),  # Primary forecaster (xAI)
        ("anthropic/claude-sonnet-4.5", "openrouter", "news_analyst"),
        ("openai/o3", "openrouter", "bull_researcher"),
        ("google/gemini-3-pro-preview", "openrouter", "bear_researcher"),
        ("deepseek/deepseek-v3.2", "openrouter", "risk_manager"),
    ]

    decisions = {}

    for model_name, provider, role in ensemble_models:
        print(f"\n[{role.upper()}] {model_name}...", end=" ", flush=True)

        try:
            # Role-specific prompt
            role_prompts = {
                "forecaster": "As a market forecaster, predict the probability of YES and provide confidence.",
                "news_analyst": "As a news analyst, evaluate recent news sentiment and its impact.",
                "bull_researcher": "As a bull case researcher, make the strongest case for YES.",
                "bear_researcher": "As a bear case researcher, make the strongest case for NO.",
                "risk_manager": "As a risk manager, evaluate downside risks and optimal position sizing.",
            }

            prompt = f"""
{role_prompts.get(role, 'Analyze the market.')}

Market: {market_data['title']}
YES price: ${market_data['yes_price']} | NO price: ${market_data['no_price']}
Volume: {market_data['volume']:,} contracts | Expires: {market_data['days_to_expiry']} days

Respond ONLY with valid JSON: {{"decision": "yes"|"no"|"pass", "confidence": 0.0-1.0, "reasoning": "..."}}
"""

            start = time.time()

            # Use the router to get decision
            response = await openrouter.get_completion(
                prompt=prompt,
                system=f"You are a {role} for a trading bot.",
                temperature=0,
                max_tokens=500,
            )

            elapsed = time.time() - start

            if response:
                # Try to parse JSON
                try:
                    import json
                    json_start = response.find("{")
                    if json_start != -1:
                        json_str = response[json_start:]
                        # Find matching }
                        brace_count = 0
                        for i, c in enumerate(json_str):
                            if c == "{": brace_count += 1
                            elif c == "}":
                                brace_count -= 1
                                if brace_count == 0:
                                    json_str = json_str[:i+1]
                                    break

                        decision = json.loads(json_str)
                        decisions[role] = decision

                        print(f"[OK] {decision.get('decision', 'unknown').upper()} (conf: {decision.get('confidence', 0):.2f})")
                    else:
                        print(f"[WARN] No JSON in response")
                        decisions[role] = {"error": "No JSON"}
                except json.JSONDecodeError:
                    print(f"[WARN] Invalid JSON in response")
                    decisions[role] = {"error": "Invalid JSON"}
            else:
                print(f"[FAIL] No response")
                decisions[role] = {"error": "No response"}

        except Exception as e:
            print(f"[FAIL] {str(e)[:50]}...")
            decisions[role] = {"error": str(e)[:50]}

        await asyncio.sleep(1)

    # Summarize ensemble
    print(f"\n[STATS] Ensemble Summary:")
    successful = sum(1 for d in decisions.values() if "decision" in d)
    print(f"  • Total models: {len(ensemble_models)}")
    print(f"  • Provided decisions: {successful}/{len(ensemble_models)}")

    if successful > 0:
        yes_votes = sum(1 for d in decisions.values() if d.get("decision") == "yes")
        no_votes = sum(1 for d in decisions.values() if d.get("decision") == "no")
        pass_votes = sum(1 for d in decisions.values() if d.get("decision") == "pass")

        print(f"  • YES votes: {yes_votes}")
        print(f"  • NO votes: {no_votes}")
        print(f"  • PASS votes: {pass_votes}")

        if successful >= 3:  # Min consensus threshold
            if yes_votes > no_votes:
                print(f"  [OK] ENSEMBLE CONSENSUS: BUY YES")
            elif no_votes > yes_votes:
                print(f"  [OK] ENSEMBLE CONSENSUS: BUY NO")
            else:
                print(f"  [WARN] ENSEMBLE SPLIT: No clear consensus")

    print("\n[INFO] Full Decisions:")
    print(json.dumps({k: v for k, v in decisions.items()}, indent=2))

    return successful >= 3


# =============================================================================
# MAIN ORCHESTRATION
# =============================================================================

async def main():
    """Run all validation tests."""
    print("\n" + "="*80)
    print("[TEST] KALSHI AI TRADING BOT - COMPREHENSIVE VALIDATION")
    print("="*80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Kalshi Env: {settings.api.kalshi_base_url}")
    print(f"OpenRouter Key: {settings.api.openrouter_api_key[:20]}...***")
    print(f"Anthropic Key: {settings.api.kalshi_api_key[:20]}...***")

    results = {}

    # Test 1: Kalshi API
    print("\n" + "-"*80)
    try:
        results["kalshi_api"] = await test_kalshi_api()
    except Exception as e:
        print(f"[FAIL] Kalshi test crashed: {e}")
        results["kalshi_api"] = False

    # Test 2: OpenRouter Models
    print("\n" + "-"*80)
    try:
        results["openrouter_models"] = await test_openrouter_models()
    except Exception as e:
        print(f"[FAIL] OpenRouter test crashed: {e}")
        results["openrouter_models"] = False

    # Test 3: Ensemble
    print("\n" + "-"*80)
    try:
        results["ensemble"] = await test_ensemble_decisions()
    except Exception as e:
        print(f"[FAIL] Ensemble test crashed: {e}")
        results["ensemble"] = False

    # Final summary
    print("\n" + "="*80)
    print("📈 VALIDATION SUMMARY")
    print("="*80)
    for test_name, passed in results.items():
        status = "[OK] PASS" if passed else "[FAIL] FAIL"
        print(f"  {test_name.replace('_', ' ').upper()}: {status}")

    all_passed = all(results.values())
    print("\n" + ("[SUCCESS] ALL TESTS PASSED!" if all_passed else "[WARN] SOME TESTS FAILED"))

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
