"""
Configuration validation and management for Book RAG Agent.

This module validates all required environment variables on startup
and provides centralized configuration access.
"""

import os
import sys
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class ConfigValidationError:
    """Represents a configuration validation error."""
    key: str
    message: str
    is_critical: bool = True


@dataclass
class AppConfig:
    """Application configuration with validated values."""

    # API Keys
    gemini_api_key: str = ""
    cohere_api_key: str = ""
    openai_api_key: str = ""

    # Qdrant
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection_name: str = "book_chunks"

    # Model Configuration
    gemini_model_name: str = "gemini-2.5-flash"
    embedding_model: str = "embed-english-v3.0"

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    log_level: str = "INFO"

    # Processing Settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_results: int = 5
    docs_directory: str = "docs"

    # CORS
    cors_origins: List[str] = field(default_factory=lambda: ["*"])

    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600


class ConfigValidator:
    """Validates and loads application configuration."""

    REQUIRED_VARS = [
        ("GEMINI_API_KEY", "Required for embeddings and chat generation"),
        ("QDRANT_URL", "Required for vector database connection"),
    ]

    OPTIONAL_VARS = [
        ("QDRANT_API_KEY", "Required for Qdrant Cloud authentication"),
        ("COHERE_API_KEY", "Optional: for Cohere embeddings"),
        ("OPENAI_API_KEY", "Optional: for OpenAI integration"),
    ]

    def __init__(self):
        self.errors: List[ConfigValidationError] = []
        self.warnings: List[str] = []
        self.config: Optional[AppConfig] = None

    def validate(self) -> bool:
        """
        Validate all configuration values.

        Returns:
            bool: True if all critical validations pass
        """
        self.errors = []
        self.warnings = []

        # Check required variables
        for var_name, description in self.REQUIRED_VARS:
            value = os.getenv(var_name)
            if not value or value.strip() == "":
                self.errors.append(ConfigValidationError(
                    key=var_name,
                    message=f"Missing required environment variable: {var_name}. {description}",
                    is_critical=True
                ))

        # Check optional variables
        for var_name, description in self.OPTIONAL_VARS:
            value = os.getenv(var_name)
            if not value or value.strip() == "":
                self.warnings.append(f"Optional variable not set: {var_name}. {description}")

        # Validate specific formats
        self._validate_qdrant_url()
        self._validate_port()
        self._validate_numeric_values()

        return len([e for e in self.errors if e.is_critical]) == 0

    def _validate_qdrant_url(self) -> None:
        """Validate Qdrant URL format."""
        url = os.getenv("QDRANT_URL", "")
        if url and not (url.startswith("http://") or url.startswith("https://")):
            self.errors.append(ConfigValidationError(
                key="QDRANT_URL",
                message=f"Invalid QDRANT_URL format: {url}. Must start with http:// or https://",
                is_critical=True
            ))

    def _validate_port(self) -> None:
        """Validate port number."""
        port_str = os.getenv("PORT", "8000")
        try:
            port = int(port_str)
            if port < 1 or port > 65535:
                self.errors.append(ConfigValidationError(
                    key="PORT",
                    message=f"Invalid PORT: {port}. Must be between 1 and 65535",
                    is_critical=False
                ))
        except ValueError:
            self.errors.append(ConfigValidationError(
                key="PORT",
                message=f"Invalid PORT: {port_str}. Must be a number",
                is_critical=False
            ))

    def _validate_numeric_values(self) -> None:
        """Validate numeric configuration values."""
        numeric_vars = [
            ("CHUNK_SIZE", 100, 10000),
            ("CHUNK_OVERLAP", 0, 1000),
            ("TOP_K_RESULTS", 1, 50),
            ("RATE_LIMIT_REQUESTS", 1, 10000),
            ("RATE_LIMIT_WINDOW", 60, 86400),
        ]

        for var_name, min_val, max_val in numeric_vars:
            value_str = os.getenv(var_name)
            if value_str:
                try:
                    value = int(value_str)
                    if value < min_val or value > max_val:
                        self.warnings.append(
                            f"{var_name}={value} is outside recommended range [{min_val}, {max_val}]"
                        )
                except ValueError:
                    self.errors.append(ConfigValidationError(
                        key=var_name,
                        message=f"Invalid {var_name}: {value_str}. Must be a number",
                        is_critical=False
                    ))

    def load_config(self) -> AppConfig:
        """
        Load and return validated configuration.

        Returns:
            AppConfig: Loaded configuration object
        """
        def safe_int(value: str, default: int) -> int:
            try:
                return int(value) if value else default
            except (ValueError, TypeError):
                return default

        def safe_bool(value: str, default: bool) -> bool:
            if not value:
                return default
            return value.lower() in ("true", "1", "yes")

        cors_origins_str = os.getenv("CORS_ORIGINS", "*")
        cors_origins = [o.strip() for o in cors_origins_str.split(",") if o.strip()]

        self.config = AppConfig(
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            cohere_api_key=os.getenv("COHERE_API_KEY", ""),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            qdrant_url=os.getenv("QDRANT_URL", ""),
            qdrant_api_key=os.getenv("QDRANT_API_KEY", ""),
            qdrant_collection_name=os.getenv("QDRANT_COLLECTION_NAME", "book_chunks"),
            gemini_model_name=os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "embed-english-v3.0"),
            host=os.getenv("HOST", "0.0.0.0"),
            port=safe_int(os.getenv("PORT"), 8000),
            debug=safe_bool(os.getenv("DEBUG"), False),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            chunk_size=safe_int(os.getenv("CHUNK_SIZE"), 1000),
            chunk_overlap=safe_int(os.getenv("CHUNK_OVERLAP"), 200),
            top_k_results=safe_int(os.getenv("TOP_K_RESULTS"), 5),
            docs_directory=os.getenv("DOCS_DIRECTORY", "docs"),
            cors_origins=cors_origins,
            rate_limit_requests=safe_int(os.getenv("RATE_LIMIT_REQUESTS"), 100),
            rate_limit_window=safe_int(os.getenv("RATE_LIMIT_WINDOW"), 3600),
        )

        return self.config

    def print_status(self) -> None:
        """Print configuration status to console."""
        print("\n" + "=" * 60)
        print("CONFIGURATION VALIDATION")
        print("=" * 60)

        if self.errors:
            print("\n[X] ERRORS:")
            for error in self.errors:
                critical = "[CRITICAL]" if error.is_critical else "[WARNING]"
                print(f"  {critical} {error.key}: {error.message}")

        if self.warnings:
            print("\n[!] WARNINGS:")
            for warning in self.warnings:
                print(f"  - {warning}")

        if not self.errors and not self.warnings:
            print("\n[OK] All configuration values are valid!")

        print("=" * 60 + "\n")


def validate_config_on_startup() -> AppConfig:
    """
    Validate configuration on application startup.

    Raises:
        ValueError: If critical configuration is missing (instead of SystemExit for serverless compatibility)

    Returns:
        AppConfig: Validated configuration
    """
    validator = ConfigValidator()
    is_valid = validator.validate()
    config = validator.load_config()

    validator.print_status()

    if not is_valid:
        error_msgs = [f"{error.key}: {error.message}" for error in validator.errors if error.is_critical]
        raise ValueError(
            f"Cannot start application due to configuration errors. "
            f"Missing or invalid: {', '.join(error_msgs)}"
        )

    return config


# Global config instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """
    Get the global configuration instance.

    Returns:
        AppConfig: Application configuration
    """
    global _config
    if _config is None:
        _config = validate_config_on_startup()
    return _config
