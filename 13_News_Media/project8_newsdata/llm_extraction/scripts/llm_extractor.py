"""
LLM-Based Extractor for Disaster Information

This module implements Large Language Model-based extraction methods
for structured disaster information from Vietnamese news articles.
"""

import json
import time
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import os

from config.llm_config import LLM_CONFIGS, EXTRACTION_SETTINGS, validate_api_keys, estimate_cost
from config.prompts import get_prompt_template, PROMPT_CONFIGS


@dataclass
class LLMExtractionResult:
    """Container for LLM extraction results"""
    extraction_id: str
    timestamp: str
    source_text: str
    extracted_info: Dict[str, Any]
    model_used: str
    provider: str
    processing_time: float
    cost_estimate: float
    confidence_score: float
    raw_response: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractionMetrics:
    """Metrics for extraction performance"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_cost: float = 0.0
    average_processing_time: float = 0.0
    average_confidence: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0


class LLMExtractor:
    """
    Large Language Model-based extractor for disaster information.

    Supports multiple LLM providers: OpenAI, Anthropic, Groq
    with automatic fallback and cost optimization.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the LLM extractor.

        Args:
            config: Optional configuration override
        """
        self.config = EXTRACTION_SETTINGS.copy()
        if config:
            self.config.update(config)

        self.logger = self._setup_logging()
        self.metrics = ExtractionMetrics()

        # Validate API keys
        self.available_models = self._validate_and_setup_models()

        # Setup caching
        self.cache = {} if self.config["cache_enabled"] else None
        self.cache_ttl = self.config["cache_ttl_hours"] * 3600  # Convert to seconds

        # Initialize LLM clients
        self.clients = self._initialize_clients()

        self.logger.info(f"Initialized LLMExtractor with {len(self.available_models)} available models")

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(__name__)
        logger.setLevel(getattr(logging, self.config["log_level"]))

        if self.config["enable_console_logging"]:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        if self.config.get("log_file"):
            file_handler = logging.FileHandler(self.config["log_file"])
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

        return logger

    def _validate_and_setup_models(self) -> List[str]:
        """Validate API keys and setup available models"""
        validation = validate_api_keys()
        available_models = []

        for model_name, config in LLM_CONFIGS.items():
            if validation.get(config.api_key_env, False):
                available_models.append(model_name)
                self.logger.info(f"Model {model_name} available ({config.provider})")
            else:
                self.logger.warning(f"Model {model_name} unavailable - missing {config.api_key_env}")

        if not available_models:
            raise ValueError("No LLM models available. Please set API keys for at least one provider.")

        return available_models

    def _initialize_clients(self) -> Dict[str, Any]:
        """Initialize LLM API clients"""
        clients = {}

        # OpenAI client
        if any(m for m in self.available_models if LLM_CONFIGS[m].provider == "openai"):
            try:
                import openai
                clients["openai"] = openai.OpenAI(
                    api_key=os.getenv("OPENAI_API_KEY")
                )
            except ImportError:
                self.logger.warning("OpenAI library not installed")

        # Anthropic client
        if any(m for m in self.available_models if LLM_CONFIGS[m].provider == "anthropic"):
            try:
                import anthropic
                clients["anthropic"] = anthropic.Anthropic(
                    api_key=os.getenv("ANTHROPIC_API_KEY")
                )
            except ImportError:
                self.logger.warning("Anthropic library not installed")

        # Groq client
        if any(m for m in self.available_models if LLM_CONFIGS[m].provider == "groq"):
            try:
                import groq
                clients["groq"] = groq.Groq(
                    api_key=os.getenv("GROQ_API_KEY")
                )
            except ImportError:
                self.logger.warning("Groq library not installed")

        return clients

    def _get_cache_key(self, text: str, model: str, prompt_type: str) -> str:
        """Generate cache key for text + model + prompt combination"""
        content = f"{text}_{model}_{prompt_type}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached result if available and not expired"""
        if not self.cache or cache_key not in self.cache:
            return None

        cached_item = self.cache[cache_key]
        if time.time() - cached_item["timestamp"] > self.cache_ttl:
            del self.cache[cache_key]
            return None

        self.metrics.cache_hits += 1
        return cached_item["result"]

    def _cache_result(self, cache_key: str, result: Dict[str, Any]):
        """Cache extraction result"""
        if self.cache is not None:
            self.cache[cache_key] = {
                "result": result,
                "timestamp": time.time()
            }

    def _select_model(self, preferred_model: Optional[str] = None) -> str:
        """Select appropriate model based on availability and preferences"""
        if preferred_model and preferred_model in self.available_models:
            return preferred_model

        # Use default model if available
        if self.config["default_model"] in self.available_models:
            return self.config["default_model"]

        # Fallback to cheapest available model
        for fallback in self.config["fallback_models"]:
            if fallback in self.available_models:
                return fallback

        # Use first available model
        return self.available_models[0]

    def _call_llm_api(self, model: str, prompt: str, **kwargs) -> Tuple[str, float]:
        """Call LLM API with error handling and retries"""
        config = LLM_CONFIGS[model]
        provider = config.provider
        client = self.clients.get(provider)

        if not client:
            raise ValueError(f"No client available for provider: {provider}")

        max_retries = kwargs.get("max_retries", config.max_retries)
        retry_delay = kwargs.get("retry_delay", config.retry_delay)

        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()

                if provider == "openai":
                    response = client.chat.completions.create(
                        model=config.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=config.max_tokens,
                        temperature=config.temperature,
                        top_p=config.top_p,
                        frequency_penalty=config.frequency_penalty,
                        presence_penalty=config.presence_penalty,
                        timeout=config.timeout
                    )
                    content = response.choices[0].message.content
                    cost = estimate_cost(len(prompt), model)

                elif provider == "anthropic":
                    response = client.messages.create(
                        model=config.model_name,
                        max_tokens=config.max_tokens,
                        temperature=config.temperature,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    content = response.content[0].text
                    cost = estimate_cost(len(prompt), model)

                elif provider == "groq":
                    response = client.chat.completions.create(
                        model=config.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=config.max_tokens,
                        temperature=config.temperature,
                        timeout=config.timeout
                    )
                    content = response.choices[0].message.content
                    cost = estimate_cost(len(prompt), model)

                processing_time = time.time() - start_time

                self.logger.debug(f"LLM call successful: {model} ({processing_time:.2f}s, ${cost:.4f})")
                return content, cost

            except Exception as e:
                self.logger.warning(f"LLM call attempt {attempt + 1} failed: {model} - {str(e)}")

                if attempt < max_retries:
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    raise e

        raise RuntimeError(f"Failed to call LLM after {max_retries + 1} attempts")

    def _parse_llm_response(self, response: str) -> Tuple[Dict[str, Any], float]:
        """Parse LLM response and extract JSON with confidence score"""
        try:
            # Try to extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1

            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")

            json_str = response[json_start:json_end]
            parsed_data = json.loads(json_str)

            # Calculate confidence based on response quality
            confidence = self._calculate_confidence(parsed_data, response)

            return parsed_data, confidence

        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"Failed to parse LLM response: {str(e)}")
            self.logger.debug(f"Raw response: {response}")

            # Return minimal fallback
            return {
                "error": "Failed to parse response",
                "raw_response": response[:500]
            }, 0.0

    def _calculate_confidence(self, data: Dict[str, Any], raw_response: str) -> float:
        """Calculate confidence score for extracted data"""
        if "error" in data:
            return 0.0

        confidence = 0.8  # Base confidence

        # Check for key fields
        key_fields = ["type", "location", "time"]
        present_fields = sum(1 for field in key_fields if data.get(field))

        if present_fields == 3:
            confidence += 0.1
        elif present_fields == 2:
            confidence += 0.05

        # Check for numeric fields
        if isinstance(data.get("deaths"), (int, float)) and data["deaths"] >= 0:
            confidence += 0.05
        if isinstance(data.get("injured"), (int, float)) and data["injured"] >= 0:
            confidence += 0.05

        # Length appropriateness
        response_length = len(raw_response)
        if 100 < response_length < 2000:
            confidence += 0.05
        elif response_length < 50:
            confidence -= 0.1

        return min(confidence, 1.0)

    def extract_disaster_info(
        self,
        text: str,
        model: Optional[str] = None,
        prompt_type: str = "full"
    ) -> LLMExtractionResult:
        """
        Extract disaster information from text using LLM.

        Args:
            text: Input text to extract from
            model: Preferred model to use
            prompt_type: Type of prompt template

        Returns:
            ExtractionResult with structured information
        """
        start_time = time.time()
        extraction_id = f"llm_extract_{int(time.time() * 1000)}_{hash(text) % 10000}"

        self.logger.info(f"Starting LLM extraction: {extraction_id}")

        # Select model
        selected_model = self._select_model(model)
        model_config = LLM_CONFIGS[selected_model]

        # Check cache
        cache_key = self._get_cache_key(text, selected_model, prompt_type)
        cached_result = self._get_cached_result(cache_key)

        if cached_result:
            self.logger.info(f"Cache hit for {extraction_id}")
            processing_time = time.time() - start_time
            return LLMExtractionResult(
                extraction_id=extraction_id,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                source_text=text,
                extracted_info=cached_result["extracted_info"],
                model_used=selected_model,
                provider=model_config.provider,
                processing_time=processing_time,
                cost_estimate=0.0,  # Cached, no cost
                confidence_score=cached_result["confidence_score"],
                raw_response=cached_result["raw_response"],
                metadata={"cached": True}
            )

        self.metrics.cache_misses += 1

        try:
            # Get prompt template
            prompt = get_prompt_template(prompt_type, text=text)

            # Call LLM
            raw_response, cost = self._call_llm_api(selected_model, prompt)

            # Parse response
            extracted_info, confidence = self._parse_llm_response(raw_response)

            # Cache result
            cache_data = {
                "extracted_info": extracted_info,
                "confidence_score": confidence,
                "raw_response": raw_response
            }
            self._cache_result(cache_key, cache_data)

            processing_time = time.time() - start_time

            # Update metrics
            self.metrics.total_requests += 1
            self.metrics.successful_requests += 1
            self.metrics.total_cost += cost
            self.metrics.average_processing_time = (
                (self.metrics.average_processing_time * (self.metrics.total_requests - 1)) +
                processing_time
            ) / self.metrics.total_requests
            self.metrics.average_confidence = (
                (self.metrics.average_confidence * (self.metrics.total_requests - 1)) +
                confidence
            ) / self.metrics.total_requests

            result = LLMExtractionResult(
                extraction_id=extraction_id,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                source_text=text,
                extracted_info=extracted_info,
                model_used=selected_model,
                provider=model_config.provider,
                processing_time=processing_time,
                cost_estimate=cost,
                confidence_score=confidence,
                raw_response=raw_response,
                metadata={
                    "prompt_type": prompt_type,
                    "text_length": len(text),
                    "cached": False
                }
            )

            self.logger.info(f"LLM extraction completed: {extraction_id} ({confidence:.2f} confidence, ${cost:.4f})")
            return result

        except Exception as e:
            self.logger.error(f"LLM extraction failed: {extraction_id} - {str(e)}")
            self.metrics.total_requests += 1
            self.metrics.failed_requests += 1

            processing_time = time.time() - start_time

            return LLMExtractionResult(
                extraction_id=extraction_id,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                source_text=text,
                extracted_info={"error": str(e)},
                model_used=selected_model,
                provider=model_config.provider,
                processing_time=processing_time,
                cost_estimate=0.0,
                confidence_score=0.0,
                raw_response="",
                metadata={"error": True, "error_message": str(e)}
            )

    def extract_from_texts(
        self,
        texts: List[str],
        model: Optional[str] = None,
        prompt_type: str = "full",
        batch_size: int = 5
    ) -> List[LLMExtractionResult]:
        """
        Extract disaster information from multiple texts.

        Args:
            texts: List of texts to process
            model: Preferred model to use
            prompt_type: Type of prompt template
            batch_size: Number of concurrent requests

        Returns:
            List of extraction results
        """
        self.logger.info(f"Starting batch extraction of {len(texts)} texts")

        results = []

        # Process in batches to respect rate limits
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1} with {len(batch_texts)} texts")

            # Process batch concurrently if async is available
            try:
                batch_results = self._process_batch_async(batch_texts, model, prompt_type)
            except ImportError:
                # Fallback to sequential processing
                batch_results = []
                for text in batch_texts:
                    result = self.extract_disaster_info(text, model, prompt_type)
                    batch_results.append(result)

            results.extend(batch_results)

            # Rate limiting
            if i + batch_size < len(texts):
                time.sleep(1)  # Brief pause between batches

        return results

    async def _process_batch_async(
        self,
        texts: List[str],
        model: Optional[str],
        prompt_type: str
    ) -> List[LLMExtractionResult]:
        """Process batch asynchronously"""
        import aiohttp
        import asyncio

        semaphore = asyncio.Semaphore(self.config["max_concurrent_requests"])
        results = []

        async def process_single(text: str):
            async with semaphore:
                # Note: This is a simplified async version
                # In practice, you'd need async LLM clients
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    self.extract_disaster_info,
                    text,
                    model,
                    prompt_type
                )
                return result

        tasks = [process_single(text) for text in texts]
        results = await asyncio.gather(*tasks)

        return results

    def save_results(self, results: List[LLMExtractionResult], output_path: str):
        """
        Save extraction results to file.

        Args:
            output_path: Path to save results
        """
        output_data = [result.to_dict() for result in results]

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Saved {len(results)} LLM extraction results to {output_path}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get extraction performance metrics"""
        return {
            "total_requests": self.metrics.total_requests,
            "successful_requests": self.metrics.successful_requests,
            "failed_requests": self.metrics.failed_requests,
            "success_rate": (
                self.metrics.successful_requests / self.metrics.total_requests
                if self.metrics.total_requests > 0 else 0
            ),
            "total_cost": self.metrics.total_cost,
            "average_processing_time": self.metrics.average_processing_time,
            "average_confidence": self.metrics.average_confidence,
            "cache_hit_rate": (
                self.metrics.cache_hits / (self.metrics.cache_hits + self.metrics.cache_misses)
                if (self.metrics.cache_hits + self.metrics.cache_misses) > 0 else 0
            ),
            "available_models": self.available_models
        }

    def validate_extraction(self, text: str, extraction: Dict[str, Any]) -> Dict[str, Any]:
        """Validate extraction quality and detect potential issues"""
        issues = []

        # Check for hallucination indicators
        if "error" in extraction:
            issues.append("parsing_error")

        # Check for unrealistic numbers
        if extraction.get("deaths", 0) > 10000:
            issues.append("unrealistic_death_count")
        if extraction.get("injured", 0) > 50000:
            issues.append("unrealistic_injury_count")

        # Check for missing key information
        if not extraction.get("type"):
            issues.append("missing_disaster_type")
        if not extraction.get("location"):
            issues.append("missing_location")

        # Check text consistency
        disaster_keywords = ["bão", "lũ", "động đất", "sạt lở", "cháy"]
        has_disaster_keyword = any(keyword in text.lower() for keyword in disaster_keywords)
        if not has_disaster_keyword:
            issues.append("no_disaster_keywords_in_text")

        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "quality_score": max(0, 1.0 - len(issues) * 0.2)
        }