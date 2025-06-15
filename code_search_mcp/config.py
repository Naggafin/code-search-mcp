from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    api_key: str = ""
    port: int = 5000
    project_path: str = str(Path.cwd())

    class Config:
        env_prefix = "MCP_"
        case_sensitive = False


settings = Settings()
