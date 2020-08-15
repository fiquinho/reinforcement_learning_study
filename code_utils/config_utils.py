import json
import logging
import shutil
from pathlib import Path


class BaseConfig(object):

    def __init__(self, config_file: Path):
        self.config_file = config_file
        self.config_dict = self.read_json_config()

    def read_json_config(self):

        with open(self.config_file, "r", encoding="utf8") as cfile:
            config_dict = json.load(cfile)

        return config_dict

    def log_configurations(self, logger: logging.Logger):

        logger.info("Used configurations:")
        for key, value in self.__dict__.items():
            if key not in ["config_file", "config_dict"]:
                logger.info(f"\t{key}: {value}")

    def copy_config(self, output_dir: Path):
        shutil.copyfile(self.config_file, output_dir / "configurations.json")
