import argparse
import asyncio
import json
import logging
import subprocess
from datetime import datetime
from functools import partial
from os import remove, environ
from pathlib import Path
from time import sleep
from typing import Optional

MAGURO_RUNNING = Path("/tmp/maguro.running")
MAGURO_FILE_PATH = Path("/tmp/maguro_never_stop")
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


def now():
    return datetime.now().strftime("%b%d-%H%M")


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


def list_task(_type):
    return sorted([_type(f.stem) for f in MAGURO_FILE_PATH.iterdir()])


def add_task(task: dict) -> None:
    tasks = list_task(int)
    if len(tasks) == 0:
        task_id = 1
    else:
        task_id = max(tasks) + 1
    task["id"] = task_id
    with (MAGURO_FILE_PATH / str(task_id)).open('w') as f:
        json.dump(task, f)


def load_task(task_id: int) -> Optional[dict]:
    next_file = MAGURO_FILE_PATH / str(task_id)
    if not next_file.exists():
        return None

    with next_file.open() as f:
        task = json.load(f)

    remove(next_file)
    return task


class Ticket(object):
    AVAILABLE_LIST = list(range(count_gpus()))

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
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None,
                                      partial(subprocess.run, command, env=env,
                                              stdout=output, stderr=subprocess.STDOUT))


def format_devices(ids: list):
    ids = [str(i) for i in ids]
    return ",".join(ids)


async def distribute_task(task_id: int):
    ticket = Ticket()
    task = load_task(task_id)
    if task is not None:
        num_gpus = task["num_gpus"]
        while ticket.is_soldout(num_gpus):
            await asyncio.sleep(1)
        gpu_ids = ticket.buy(num_gpus)
        current_env = environ.copy()
        command = task["command"]
        logger.info(f"start (gpu={format_devices(gpu_ids):>2}): {command}")
        current_env["CUDA_VISIBLE_DEVICES"] = format_devices(gpu_ids)
        current_env.update(task["env"])
        log_dir = Path(task["log_dir"])
        _now = now()
        if not log_dir.exists():
            log_dir.mkdir(parents=True)
        with (log_dir / f"{_now}-{task['id']}.log").open('w') as log_f:
            log_f.write(f"maguro {_now}\n{command}\n{'-' * 10}\n\n")
            log_f.flush()
            await run_task(command.split(), env=current_env, output=log_f)
        ticket.sell(gpu_ids)
        logger.info(COLOR.colored_str(f"finish: {command}", COLOR.GREEN))


async def daemon():
    queue = asyncio.Queue()
    queued_tasks = set()
    for t in list_task(int):
        queue.put_nowait(t)
        queued_tasks.add(t)

    async def consume():
        while not queue.empty():
            t = await queue.get()
            queued_tasks.remove(t)
            await distribute_task(t)
            new_tasks = set(list_task(int)) - queued_tasks
            if len(new_tasks) > 0:
                for t in new_tasks:
                    queue.put_nowait(t)
                    queued_tasks.add(t)
                logger.info(f"{len(new_tasks)} tasks are added")

    await asyncio.wait([consume() for _ in range(count_gpus())])


class Command(object):

    @staticmethod
    def _run(args):
        active = True
        MAGURO_RUNNING.touch()

        try:
            while active:
                # check if there are tasks every 5 seconds
                if len(list_task(int)) > 0:
                    asyncio.run(daemon())
                else:
                    sleep(5)
                active = args.forever  # if not forever, while loop finishes
            logger.info(COLOR.colored_str("finish all tasks", COLOR.GREEN))
        except KeyboardInterrupt:
            num_active = count_gpus() - Ticket.num_tickets()
            logger.warning(f"Keyboard Interrupted! {num_active} tasks might not finish properly.")
        finally:
            remove(MAGURO_RUNNING)

    @staticmethod
    def run(args):
        if MAGURO_RUNNING.exists():
            logger.warning(COLOR.colored_str("maguro is already running < ﾟ )))>< ﾟ )))><", COLOR.RED))
        elif len(list_task(str)) == 0 and not args.forever:
            logger.warning(COLOR.colored_str("No task remaining", COLOR.RED))
        else:
            Command._run(args)

    @staticmethod
    def push(args):
        exec = args.exec
        script = args.script
        command = " ".join(args.args)
        if len(exec) == 0 or len(script) == 0:
            raise RuntimeError("Need to specify exec and script (like `python train.py`)")
        script = Path(script).absolute()
        command = f"{exec} {script} {command}"
        current_env = environ.copy()
        if current_env.get("CUDA_VISIBLE_DEVICES") is not None:
            current_env.pop("CUDA_VISIBLE_DEVICES")
        task_config = {"num_gpus": args.num_gpus,
                       "command": command,
                       "log_dir": args.log_dir,
                       "env": current_env}
        for _ in range(args.num_repeat):
            add_task(task_config)
            logger.info(COLOR.colored_str(f"push: {command}", COLOR.BLUE))

    @staticmethod
    def list(args):
        tasks = list_task(str)
        suffix = "s" if len(tasks) > 1 else ""
        logger.info(f"maguro has {len(tasks)} remaining task{suffix}")
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
    p_run.add_argument("--forever", action="store_true")
    p_run.set_defaults(func=Command.run)

    p_push = sub.add_parser("push")
    p_push.add_argument("--num_repeat", "-r", type=int, default=1)
    p_push.add_argument("--num_gpus", "-g", type=int, default=1)
    p_push.add_argument("--log_dir", default="maglog")
    p_push.add_argument("exec")
    p_push.add_argument("script")
    p_push.add_argument("args", nargs=argparse.REMAINDER)
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
