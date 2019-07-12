import asyncio
import os as python_os
import pathlib
import subprocess
from functools import partial

from .utils import NOW, logger, COLOR, TicketSeller, parse_args, devices


async def run(command,
              env,
              output):
    # make subprocess.run async
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(subprocess.run, command, env=env,
                                                    stdout=output, stderr=subprocess.STDOUT))


async def distribute_trial(command,
                           trial_id,
                           args):
    ticket = TicketSeller()
    while ticket.is_soldout(args.num_gpu_per_trial):
        await asyncio.sleep(10)
    gpu_ids = ticket.buy(args.num_gpu_per_trial)
    current_env = python_os.environ.copy()
    cvd = f"CUDA_VISIBLE_DEVICES={devices(gpu_ids)}"
    if args.dryrun:
        logger.info(f"dryrun: {cvd} {command}")
        await asyncio.sleep(2)
    else:
        logger.info(f"start: {cvd} {command}")
        current_env["CUDA_VISIBLE_DEVICES"] = devices(gpu_ids)
        with (pathlib.Path(args.log_dir) / f"{NOW}-{trial_id:0>4}.log").open('w') as log_file:
            log_file.write(f"maguro {NOW}\n{command}\n{'-' * 10}\n\n")
            log_file.flush()
            await run(command.split(), env=current_env, output=log_file)
    ticket.sell(gpu_ids)
    logger.info(COLOR.colored_str(f"finish: {cvd} {command}", COLOR.GREEN))


async def _main(commands,
                args):
    await asyncio.gather(*[distribute_trial(command, trial_id, args)
                           for trial_id, command in enumerate(commands)])


def main():
    asyncio.run(_main(*parse_args()))


if __name__ == '__main__':
    main()
