from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    WEATHER_API_KEY: str
    WEATHER_BASE_URL: str

    DB_CONNECTION_STRING: str
    DB_NAME: str


@lru_cache
def get_settings() -> Settings:
    return Settings()
