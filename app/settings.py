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
    project_budget_cny: float = 500.0
    daily_budget_cny: float = 20.0
    siliconflow_api_key: str = ""
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"
    siliconflow_image_model: str = "Kwai-Kolors/Kolors"
    image_2d_backend: str = "siliconflow"
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/api/v1"
    dashscope_image_path: str = "/services/aigc/text2image/image-synthesis"
    dashscope_image_model: str = "wanx2.1-t2i-turbo"
    enable_critique: bool = True
    max_critique_step: int = 2
    enable_mcts: bool = False

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

    @property
    def chroma_dir(self) -> Path:
        d = self.data_dir / "chroma"
        d.mkdir(parents=True, exist_ok=True)
        return d


settings = Settings()
