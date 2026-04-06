"""
Anthropic Claude client for prediction market trading decisions.

Routes all AI requests through Claude via the Anthropic SDK.
Provides the same interface as OpenRouterClient for drop-in compatibility.
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from anthropic import Anthropic, AsyncAnthropic
from json_repair import repair_json

from src.config.settings import settings
from src.utils.logging_setup import TradingLoggerMixin, log_error_with_context


# ---------------------------------------------------------------------------
# Claude model pricing (USD per 1K tokens)
# ---------------------------------------------------------------------------

CLAUDE_PRICING = {
    "claude-haiku-4-5-20251001": {
        "input_per_1k": 0.00080,
        "output_per_1k": 0.004,
    },
    "claude-opus-4-6": {
        "input_per_1k": 0.015,
        "output_per_1k": 0.075,
    },
    "claude-sonnet-4-6": {
        "input_per_1k": 0.003,
        "output_per_1k": 0.015,
    },
}


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------

@dataclass
class DailyUsageTracker:
    """Tracks daily Claude API usage and cost."""
    date: str  # YYYY-MM-DD
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    request_count: int = 0
    error_count: int = 0

    def add_request(self, input_tokens: int, output_tokens: int, cost: float):
        """Log a single request."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += cost
        self.request_count += 1

    def add_error(self):
        """Log an error."""
        self.error_count += 1


@dataclass
class ModelCostTracker:
    """Accumulated cost data for Claude model."""
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost: float = 0.0
    request_count: int = 0
    error_count: int = 0
    last_used: Optional[datetime] = None


# ---------------------------------------------------------------------------
# AnthropicClient
# ---------------------------------------------------------------------------

class AnthropicClient(TradingLoggerMixin):
    """
    Async client that uses Claude via Anthropic SDK for market analysis.

    Provides the same interface as OpenRouterClient and XAIClient so callers
    can swap providers transparently.

    Features:
        * Cost tracking with Claude-specific pricing
        * Exponential-backoff retry logic
        * Temperature=0 for reproducibility
        * Structured JSON response parsing
    """

    MAX_RETRIES: int = 3
    BASE_BACKOFF: float = 1.0
    MAX_BACKOFF: float = 30.0

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the Anthropic client.

        Args:
            api_key: Anthropic API key. If not provided, uses ANTHROPIC_API_KEY env var.
            model: Claude model to use. Defaults to claude-haiku-4-5-20251001 (cheapest).
        """
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not provided and not found in environment variables"
            )

        self._model = model or "claude-haiku-4-5-20251001"

        # Initialize both sync and async clients
        self._client = Anthropic(api_key=self._api_key)
        self._async_client = AsyncAnthropic(api_key=self._api_key)

        # Cost tracking
        self._cost_tracker = ModelCostTracker(model=self._model)
        self._daily_usage: Dict[str, DailyUsageTracker] = {}

        self.logger.info(
            "Anthropic client initialized",
            model=self._model,
            api_key_present=bool(self._api_key),
        )

    # -----------------------------------------------------------------------
    # Public interface (same as OpenRouterClient)
    # -----------------------------------------------------------------------

    async def get_completion(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0,
        max_tokens: int = 2048,
    ) -> Optional[str]:
        """
        Get a text completion from Claude.

        Args:
            prompt: The user message/prompt
            system: System prompt / instructions
            temperature: Sampling temperature (0 = deterministic)
            max_tokens: Maximum output tokens

        Returns:
            The model's response text, or None on failure.
        """
        return await self._call_claude_with_retry(
            user_message=prompt,
            system_prompt=system,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def get_trading_decision(
        self,
        market_data: dict,
        context: dict = None,
        system_prompt: str = "",
    ) -> Dict[str, Any]:
        """
        Get a structured trading decision from Claude (for compatibility with ensemble).

        Args:
            market_data: Market information dict
            context: Additional context
            system_prompt: Custom system prompt

        Returns:
            Dict with decision, confidence, reasoning, etc.
        """
        # Format market into a prompt
        prompt = self._format_market_prompt(market_data)

        response_text = await self.get_completion(
            prompt=prompt,
            system=system_prompt or self._default_system_prompt(),
            temperature=0,
            max_tokens=1024,
        )

        if not response_text:
            return {"error": "No response from Claude", "decision": "pass"}

        # Try to extract JSON
        try:
            # Look for JSON code block
            json_match = response_text.find("{")
            if json_match != -1:
                json_str = response_text[json_match:]
                # Find closing brace
                brace_count = 0
                for i, char in enumerate(json_str):
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            json_str = json_str[: i + 1]
                            break

                result = json.loads(json_str)
            else:
                result = {"error": "No JSON found in response", "decision": "pass"}
        except Exception as e:
            self.logger.warning(
                "Failed to parse Claude response as JSON",
                error=str(e),
                response_snippet=response_text[:200],
            )
            result = {"error": str(e), "decision": "pass"}

        return result

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    async def _call_claude_with_retry(
        self,
        user_message: str,
        system_prompt: str = "",
        temperature: float = 0,
        max_tokens: int = 2048,
    ) -> Optional[str]:
        """
        Call Claude API with exponential backoff retry.

        Returns:
            Model response text, or None on failure after retries.
        """
        for attempt in range(self.MAX_RETRIES):
            try:
                start_time = time.time()

                response = await self._async_client.messages.create(
                    model=self._model,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=[
                        {
                            "role": "user",
                            "content": user_message,
                        }
                    ],
                    temperature=temperature,
                )

                elapsed = time.time() - start_time

                # Extract text from response
                text = response.content[0].text

                # Track cost
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                cost = self._calculate_cost(input_tokens, output_tokens)

                self._cost_tracker.input_tokens += input_tokens
                self._cost_tracker.output_tokens += output_tokens
                self._cost_tracker.total_cost += cost
                self._cost_tracker.request_count += 1
                self._cost_tracker.last_used = datetime.now()

                # Track daily usage
                today = datetime.now().strftime("%Y-%m-%d")
                if today not in self._daily_usage:
                    self._daily_usage[today] = DailyUsageTracker(date=today)
                self._daily_usage[today].add_request(input_tokens, output_tokens, cost)

                self.logger.debug(
                    "Claude completion successful",
                    model=self._model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost=round(cost, 6),
                    elapsed_seconds=round(elapsed, 2),
                )

                return text

            except Exception as e:
                self.logger.warning(
                    "Claude API call failed, retrying",
                    attempt=attempt + 1,
                    max_retries=self.MAX_RETRIES,
                    error=str(e),
                )
                self._cost_tracker.error_count += 1

                if attempt < self.MAX_RETRIES - 1:
                    # Exponential backoff
                    backoff = min(
                        self.BASE_BACKOFF * (2 ** attempt),
                        self.MAX_BACKOFF,
                    )
                    await asyncio.sleep(backoff)
                else:
                    self.logger.error(
                        "Claude API exhausted retries",
                        model=self._model,
                        error=str(e),
                    )
                    return None

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in USD for a single request."""
        pricing = CLAUDE_PRICING.get(
            self._model,
            CLAUDE_PRICING["claude-haiku-4-5-20251001"],  # fallback
        )
        input_cost = (input_tokens / 1000) * pricing["input_per_1k"]
        output_cost = (output_tokens / 1000) * pricing["output_per_1k"]
        return input_cost + output_cost

    def _format_market_prompt(self, market_data: dict) -> str:
        """Format market data into a readable prompt."""
        title = market_data.get("title", "Unknown market")
        yes_price = market_data.get("yes_price", 0.5)
        no_price = market_data.get("no_price", 0.5)
        volume = market_data.get("volume", 0)
        expiry = market_data.get("days_to_expiry", "unknown")

        return (
            f"Market: {title}\n"
            f"YES Price: ${yes_price:.2f}\n"
            f"NO Price: ${no_price:.2f}\n"
            f"Volume: {volume:,} contracts\n"
            f"Expires in: {expiry} days\n\n"
            f"Provide your prediction as a JSON object with keys: "
            f"decision (buy_yes/buy_no/pass), confidence (0-1), reasoning (str)."
        )

    def _default_system_prompt(self) -> str:
        """Default system prompt for trading analysis."""
        return (
            "You are an expert prediction market analyst. "
            "Analyze the given market data and provide a trading decision. "
            "Be concise and data-driven. "
            "Return your response as a JSON object."
        )

    # -----------------------------------------------------------------------
    # Compatibility methods
    # -----------------------------------------------------------------------

    def get_daily_cost(self, date: str = None) -> float:
        """Get daily API cost for a specific date (YYYY-MM-DD)."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        return self._daily_usage.get(date, DailyUsageTracker(date=date)).total_cost

    def get_total_cost(self) -> float:
        """Get cumulative cost across all requests."""
        return self._cost_tracker.total_cost

    def get_request_count(self) -> int:
        """Get total number of requests."""
        return self._cost_tracker.request_count

    def get_error_count(self) -> int:
        """Get total number of errors."""
        return self._cost_tracker.error_count

    def get_usage_summary(self) -> Dict[str, Any]:
        """Get a summary of usage and costs."""
        return {
            "model": self._model,
            "total_requests": self._cost_tracker.request_count,
            "total_errors": self._cost_tracker.error_count,
            "input_tokens": self._cost_tracker.input_tokens,
            "output_tokens": self._cost_tracker.output_tokens,
            "total_cost": round(self._cost_tracker.total_cost, 4),
            "daily_costs": {
                date: round(tracker.total_cost, 4)
                for date, tracker in self._daily_usage.items()
            },
        }
