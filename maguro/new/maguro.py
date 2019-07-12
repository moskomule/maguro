import argparse
import asyncio
import json
import logging
import subprocess
from datetime import datetime
from functools import partial
from os import remove, environ
from pathlib import Path

NOW = datetime.now().strftime("%b%d-%H%M")

MAGURO_FILE_PATH = Path("~/.maguro").expanduser()
if not MAGURO_FILE_PATH.exists():
    MAGURO_FILE_PATH.mkdir()

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


def task_list(_type):
    return sorted([_type(f.stem) for f in MAGURO_FILE_PATH.iterdir()])


def add_task(task: dict,
             high_priority: bool) -> None:
    tasks = task_list(int)
    if len(tasks) == 0:
        task_id = 1
    else:
        task_id = min(tasks) - 1 if high_priority else max(tasks) + 1
    task["id"] = task_id
    with (MAGURO_FILE_PATH / str(task_id)).open('w') as f:
        json.dump(task, f)


def load_task() -> dict:
    tasks = task_list(int)
    next_file = MAGURO_FILE_PATH / str(min(tasks))
    with (next_file).open() as f:
        task = json.load(f)

    remove(next_file)
    return task


def devices(ids: list):
    ids = [str(i) for i in ids]
    return ",".join(ids)


class Ticket(object):
    AVAILABLE_LIST = list(range(count_gpus()))
    RESERVED_LIST = []

    @classmethod
    def buy(cls,
            num: int):
        ids = cls.AVAILABLE_LIST[-num:]
        cls.AVAILABLE_LIST = cls.AVAILABLE_LIST[:-num]
        return ids

    @classmethod
    def sell(cls,
             ids: list):
        cls.AVAILABLE_LIST += ids

    @classmethod
    def is_soldout(cls,
                   num: int):
        return len(cls.AVAILABLE_LIST) < num

    @classmethod
    def num_tickets(cls):
        return len(cls.AVAILABLE_LIST)


async def run_task(command: list,
                   env: dict,
                   output):
    # loop = asyncio.new_event_loop()
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None,
                                      partial(subprocess.run, command, env=env,
                                              stdout=output, stderr=subprocess.STDOUT))


async def distribute_task():
    ticket = Ticket()
    while ticket.is_soldout(1):
        await asyncio.sleep(1)
    task = load_task()
    num_gpus = task["num_gpus"]
    while ticket.is_soldout(num_gpus):
        await asyncio.sleep(1)
    gpu_ids = ticket.buy(num_gpus)
    current_env = environ.copy()
    command = task["command"]
    logger.info(f"start (gpu={devices(gpu_ids):>2}): {command}")
    current_env["CUDA_VISIBLE_DEVICES"] = devices(gpu_ids)
    log_dir = Path(task["log_dir"])
    if not log_dir.exists():
        log_dir.mkdir(parents=True)
    with (log_dir / f"{NOW}-{task['id']}.log").open('w') as log_f:
        log_f.write(f"maguro {NOW}\n{command}\n{'-' * 10}\n\n")
        log_f.flush()
        await run_task(command.split(), env=current_env, output=log_f)
    ticket.sell(gpu_ids)
    logger.info(COLOR.colored_str(f"finish: {command}", COLOR.GREEN))


async def daemon(loop):
    task_history = []
    running_tasks = []
    is_running = True
    num_gpus = count_gpus()

    while len(list(MAGURO_FILE_PATH.iterdir())) > 0 and is_running:
        if len(running_tasks) <= num_gpus:
            task = loop.create_task(distribute_task())
            task_history.append(task)
            await asyncio.sleep(0)

        running_tasks = [t for t in task_history if not t.done()]
        is_running = len(running_tasks) > 0
    return [t.result() for t in task_history]


class Command(object):
    @staticmethod
    def run(args):
        MAGURO_RUNNING = Path("/tmp/maguro.running")
        if MAGURO_RUNNING.exists():
            logger.warning(COLOR.colored_str("maguro is already running", COLOR.RED))
        elif len(task_list(str)) == 0:
            logger.warning(COLOR.colored_str("No task remaining", COLOR.RED))
        else:
            MAGURO_RUNNING.touch()
            try:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(daemon(loop))
                loop.close()
                logger.info(COLOR.colored_str("finish all tasks", COLOR.GREEN))
            finally:
                remove(MAGURO_RUNNING)

    @staticmethod
    def push(args):
        command = " ".join(args.command)
        task = {"num_gpus": args.num_gpus,
                "command": command,
                "log_dir": args.log_dir}
        for _ in range(args.num_repeat):
            add_task(task, high_priority=args.high_priority)
            logger.info(COLOR.colored_str(f"push: {command}", COLOR.BLUE))

    @staticmethod
    def list(args):
        tasks = [f.stem for f in MAGURO_FILE_PATH.iterdir()]
        logger.info(f"maguro has {len(tasks)} tasks")
        if args.all:
            for t in tasks:
                with (MAGURO_FILE_PATH / t).open() as f:
                    task = json.load(f)
                    logger.info(f"{int(t):>5}: {task['command']}")

    @staticmethod
    def delete(args):
        tasks = [f.stem for f in MAGURO_FILE_PATH.iterdir()]
        ids = args.ids
        for i in ids:
            if i in tasks:
                logger.info(COLOR.colored_str(f"delete: task {i}", COLOR.GREEN))
                remove(MAGURO_FILE_PATH / i)
            else:
                logger.warning(COLOR.colored_str(f"No such task {i} to be deleted", COLOR.RED))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers()
    p_run = sub.add_parser("run")
    p_run.set_defaults(func=Command.run)

    p_push = sub.add_parser("push")
    p_push.add_argument("--num_repeat", type=int, default=1)
    p_push.add_argument("--num_gpus", type=int, default=1)
    p_push.add_argument("--log_dir", default="maglog")
    p_push.add_argument("--high_priority", action="store_true")
    p_push.add_argument("command", nargs=argparse.REMAINDER)
    p_push.set_defaults(func=Command.push)

    p_list = sub.add_parser("list")
    p_list.add_argument("--all", action="store_true")
    p_list.set_defaults(func=Command.list)

    p_del = sub.add_parser("delete")
    p_del.add_argument("ids", nargs="+")
    p_del.set_defaults(func=Command.delete)

    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
