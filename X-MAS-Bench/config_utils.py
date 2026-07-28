import json
import os
import re


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env(value):
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match):
        name = match.group(1)
        if name not in os.environ:
            raise ValueError(f"Required environment variable {name} is not set")
        return os.environ[name]

    return _ENV_PATTERN.sub(replace, value)


def load_config(path):
    with open(path, "r", encoding="utf-8") as config_file:
        config = _expand_env(json.load(config_file))
    if "model_dict" not in config:
        raise ValueError(f"Config {path} must contain a model_dict object")
    return config

