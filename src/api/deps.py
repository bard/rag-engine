from ..config import Config


def get_config() -> Config:
    return Config.from_env()
