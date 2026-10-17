import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    llama_server_url: str = field(
        default_factory=lambda: os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8999")
    )
    gateway_host: str = field(
        default_factory=lambda: os.getenv("RUBICK_HOST", "0.0.0.0")
    )
    gateway_port: int = field(
        default_factory=lambda: int(os.getenv("RUBICK_PORT", "8000"))
    )
    auto_guardian: bool = field(
        default_factory=lambda: os.getenv("RUBICK_AUTO_GUARDIAN", "false").lower() in ("true", "1", "yes")
    )
    embedding_prompt_template: str = field(
        default_factory=lambda: os.getenv(
            "RUBICK_EMBED_PROMPT", "Text: {text}\nSummary in one word:"
        )
    )
    timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("RUBICK_TIMEOUT", "60.0"))
    )


settings = Settings()
