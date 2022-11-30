from helpsk.utility import open_yaml

CONFIG = open_yaml('source/config/config.yaml')


def dir_ouput():
    return CONFIG['OUTPUT']['DIRECTORY']


def dir_data_raw():
    return CONFIG['DATA']['RAW_DIRECTORY']


def dir_data_interim():
    return CONFIG['DATA']['INTERIM_DIRECTORY']


def dir_data_external():
    return CONFIG['DATA']['EXTERNAL_DIRECTORY']


def dir_data_processed():
    return CONFIG['DATA']['PROCESSED_DIRECTORY']


def dir_notebooks():
    return CONFIG['NOTEBOOKS']['DIRECTORY']
