import copy


def update_configs(cfg_basic:dict,  cfg_update: dict):
    for key, value in cfg_update.items():
        if key in cfg_basic and isinstance(cfg_basic[key], dict):
            cfg_basic[key] = update_configs(cfg_basic[key], value)
        else:
            # deepcopy
            cfg_basic[key] = copy.deepcopy(value)
    return cfg_basic


def print_configs(cfg: dict, prefix=""):
    for key, value in cfg.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            print_configs(value, "    " + prefix)
        else:
            print(f"{prefix}{key}: {value}")
