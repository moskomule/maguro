import subprocess
import logging
import pathlib
from datetime import datetime
import argparse

NOW = datetime.now().strftime("%b%d-%H%M")


logger = logging.getLogger("maguro")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '[%(name)s|%(asctime)s] %(message)s', datefmt='%m-%d %H:%M:%S')
ch.setFormatter(formatter)
logger.addHandler(ch)


def count_gpus():
    try:
        num = len(subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True).stdout.decode('ascii').strip().split("\n"))
        return num
    except Exception as e:
        raise e


class COLOR:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    PURPLE = '\033[35m'
    CYAN = '\033[36m'
    END = '\033[0m'

    @classmethod
    def colored_str(cls,
                    s: str,
                    color: str):
        return color + s + cls.END


class TicketSeller(object):
    tickets = list(range(count_gpus()))

    @classmethod
    def buy(cls,
            num: int):

        ids = cls.tickets[-num:]
        cls.tickets = cls.tickets[:-num]
        return ids

    @classmethod
    def sell(cls,
             ids: list):
        cls.tickets += ids

    @classmethod
    def is_soldout(cls,
                   num: int):
        return len(cls.tickets) < num

    @classmethod
    def num_tickets(cls):
        return len(cls.tickets)


def read_commands(path: str):
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    with path.open() as f:
        raw_commands = f.read()

    return raw_commands.strip().split("\n")


def devices(ids: list):
    ids = [str(i) for i in ids]
    return ",".join(ids)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("commands", help="a file of a list of commands")
    p.add_argument("--num_repeat", "-r", type=int, default=1)
    p.add_argument("--dryrun", action="store_true")
    p.add_argument("--log_dir", default="magulog")
    p.add_argument("--total_gpus", type=int, default=-1)
    p.add_argument("--num_gpu_per_trial", "-n", type=int, default=1)
    args = p.parse_args()

    commands = read_commands(args.commands)
    num_repeat = args.num_repeat
    commands = commands * num_repeat

    num_gpus = TicketSeller.num_tickets()
    min_gpus = args.num_gpu_per_trial
    if num_gpus < min_gpus:
        raise RuntimeError(COLOR.colored_str(
            f"maguro requires at least {min_gpus} GPUs "
            f"but only {num_gpus} GPUs are available", COLOR.RED))
    pathlib.Path(args.log_dir).mkdir(exist_ok=True)
    logger.info(
        COLOR.colored_str(f"Total: {len(commands)} trials", COLOR.GREEN))

    if args.total_gpus == -1:
        args.total_gpus = num_gpus

    if args.total_gpus > num_gpus:
        logger.info(f"The total number of GPUs is {num_gpus}")
    else:
        for _ in range(num_gpus-args.total_gpus):
            TicketSeller.tickets.pop()

    logger.info(COLOR.colored_str(f"Resource: {num_gpus} GPUs "
                                  f"({args.num_gpu_per_trial} GPUs per trial)", COLOR.GREEN))
    return commands, args
