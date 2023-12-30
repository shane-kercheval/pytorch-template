from source.config import config  # noqa: D100


def assert_string_not_empty(value: str):  # noqa: ANN201, D103
    assert isinstance(value, str)
    assert value is not None
    assert value.strip() != ''


def test_config_values_are_not_empty() -> None:  # noqa: D103
    assert_string_not_empty(config.dir_data_processed())
    assert_string_not_empty(config.dir_ouput())
    assert_string_not_empty(config.dir_data_raw())
    assert_string_not_empty(config.dir_data_interim())
    assert_string_not_empty(config.dir_data_external())
    assert_string_not_empty(config.dir_data_processed())
    assert_string_not_empty(config.dir_notebooks())
