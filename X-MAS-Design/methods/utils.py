import yaml


class ProtocolParseError(ValueError):
    """Raised when a model response violates a required MAS protocol."""

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def handle_retry_error(retry_state):
    raise retry_state.outcome.exception()
