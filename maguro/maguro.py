import pathlib
import subprocess
import argparse
import asyncio
import logging
import os as python_os
from functools import partial

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
        logger.info(COLOR.GREEN + f"Use {num} GPUs" + COLOR.END)
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


async def run(command, env):
    # make subprocess.run async
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(subprocess.run, command, env=env))


class Resource(object):
    tickets = list(range(count_gpus()))

    @staticmethod
    def buy():
        if len(Resource.tickets) == 0:
            return None
        else:
            return Resource.tickets.pop()

    @staticmethod
    def sell(id):
        Resource.tickets.append(id)

    @staticmethod
    def has_ticket():
        return len(Resource.tickets) != 0


async def distribute_job(command, dryrun=False):
    r = Resource()
    while not r.has_ticket():
        await asyncio.sleep(10)
    gpu_id = r.buy()
    current_env = python_os.environ.copy()
    if dryrun:
        logger.info(f"dryrun: CUDA_VISIBLE_DEVICES={gpu_id} {command}")
        await asyncio.sleep(2)
    else:
        logger.info(f"start: CUDA_VISIBLE_DEVICES={gpu_id} {command}")
        current_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        await run(command.split(), env=current_env)
    r.sell(gpu_id)
    logger.info(
        COLOR.BLUE + f"finish: CUDA_VISIBLE_DEVICES={gpu_id} {command}" + COLOR.END)


async def _main(commands, dryrun):
    await asyncio.gather(*[distribute_job(command, dryrun) for command in commands])


def read_commands(path: str):
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    with path.open() as f:
        raw_commands = f.read()

    return raw_commands.strip().split("\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("commands", help="a file of a list of commands")
    p.add_argument("--num_repeat", "-n", type=int, default=1)
    p.add_argument("--dryrun", action="store_true")
    args = p.parse_args()

    commands = read_commands(args.commands)
    num_repeat = args.num_repeat
    commands = commands * num_repeat
    dryrun = args.dryrun
    logger.info(COLOR.GREEN + f"Total: {len(commands)} trials" + COLOR.END)
    asyncio.run(_main(commands, dryrun))


if __name__ == '__main__':
    main()
