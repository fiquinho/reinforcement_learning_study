import logging


FORMATTER = logging.Formatter('%(asctime)s: - [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')


def prepare_stream_logger(logger: logging.Logger, level: int=logging.INFO) -> None:
    """
    Configure a logger to print to the console.
    :param logger: The Logger object to configure.
    :param level: The threshold of the logger.
    :return:
    """
    logger.setLevel(level)

    formatter = FORMATTER
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)