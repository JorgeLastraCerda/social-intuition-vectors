from src.utils.config import load_config


def test_load_config_default_path():
    config = load_config("config/config.yaml")
    assert config.model.name == "google/gemma-3-12b-it"
    assert config.model.backend == "transformer-bridge"
    assert config.probing.probe_layer_frac == 0.66
    assert config.paths.results.as_posix() == "results"
