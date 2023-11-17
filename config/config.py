from configparser import ConfigParser


def parse_config(path='config/config.ini'):

    config_parser = ConfigParser()

    try:
        with open(path, 'r') as config_file:
            config_parser.read_file(config_file)

    except FileNotFoundError:
        print(f'Error: could not find config file: {path}')
        exit(1)

    return config_parser


config = parse_config()


def get(section: str, option: str) -> str:
    return config.get(section, option)


def get_int(section: str, option: str) -> int:
    return config.getint(section, option)


def get_float(section: str, option: str) -> float:
    return config.getfloat(section, option)


def get_default_bool(section: str, option: str, default: bool) -> bool:
    try:
        return config.getboolean(section, option, fallback=default)
    except ValueError:
        return default


def get_section(section: str):
    return dict(config.items(section))
