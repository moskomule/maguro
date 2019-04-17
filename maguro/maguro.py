import pathlib
import subprocess
import argparse
import asyncio
import logging
import os as python_os
from functools import partial
from datetime import date

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
        if num == 0:
            raise RuntimeError("maguro requires at least 1 GPU")
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


async def run(command, env, output):
    # make subprocess.run async
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(subprocess.run, command, env=env,
                                                    stdout=output, stderr=subprocess.STDOUT))


class TicketSeller(object):
    tickets = list(range(count_gpus()))

    @staticmethod
    def buy():
        if len(TicketSeller.tickets) == 0:
            return None
        else:
            return TicketSeller.tickets.pop()

    @staticmethod
    def sell(id):
        TicketSeller.tickets.append(id)

    @staticmethod
    def is_soldout():
        return len(TicketSeller.tickets) != 0


async def distribute_job(command, job_id, args):
    ticket = TicketSeller()
    while ticket.is_soldout():
        await asyncio.sleep(10)
    gpu_id = ticket.buy()
    current_env = python_os.environ.copy()
    if args.dryrun:
        logger.info(f"dryrun: CUDA_VISIBLE_DEVICES={gpu_id} {command}")
        await asyncio.sleep(2)
    else:
        logger.info(f"start: CUDA_VISIBLE_DEVICES={gpu_id} {command}")
        current_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        with (pathlib.Path(args.log_dir) / f"{NOW}-{job_id:0>4}.log").open('w') as f:
            await run(command.split(), env=current_env, output=f)
    ticket.sell(gpu_id)
    logger.info(
        COLOR.BLUE + f"finish: CUDA_VISIBLE_DEVICES={gpu_id} {command}" + COLOR.END)


async def _main(commands, args):
    await asyncio.gather(*[distribute_job(command, job_id, args)
                           for job_id, command in enumerate(commands)])


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
    p.add_argument("--log_dir", default="magulog")
    args = p.parse_args()

    commands = read_commands(args.commands)
    num_repeat = args.num_repeat
    commands = commands * num_repeat
    pathlib.Path(args.log_dir).mkdir(exist_ok=True)
    logger.info(COLOR.GREEN + f"Total: {len(commands)} trials" + COLOR.END)
    asyncio.run(_main(commands, args))


if __name__ == '__main__':
    main()
