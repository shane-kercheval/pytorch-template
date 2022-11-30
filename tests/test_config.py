from source.config import config


def assert_string_not_empty(value: str):
    assert isinstance(value, str)
    assert value is not None
    assert value.strip() != ''


def test_config_values_are_not_empty():
    assert_string_not_empty(config.dir_data_processed())
    assert_string_not_empty(config.dir_ouput())
    assert_string_not_empty(config.dir_data_raw())
    assert_string_not_empty(config.dir_data_interim())
    assert_string_not_empty(config.dir_data_external())
    assert_string_not_empty(config.dir_data_processed())
    assert_string_not_empty(config.dir_notebooks())
    assert_string_not_empty(config.experiment_server_url())
    assert_string_not_empty(config.experiment_client_url())
    assert_string_not_empty(config.experiment_name())
