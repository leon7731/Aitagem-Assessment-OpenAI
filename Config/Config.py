# from pydantic import BaseSettings
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    langchain_api_key: str
    langchain_project: str
    openai_api_key: str

    model_config = SettingsConfigDict(env_file="Config/.env", env_file_encoding='utf-8')


settings = Settings()

