from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    styleforge_data_dir: Path = Path("./data")
    styleforge_host: str = "127.0.0.1"
    styleforge_port: int = 8000
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"

    @property
    def data_dir(self) -> Path:
        p = self.styleforge_data_dir
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def assets_dir(self) -> Path:
        d = self.data_dir / "assets"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def jobs_db(self) -> Path:
        return self.data_dir / "jobs.db"


settings = Settings()
