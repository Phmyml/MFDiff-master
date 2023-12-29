"""
Train a diffusion model on images.
"""
import sys

print(sys.executable)
import argparse
sys.path.append("..")
sys.path.append(".")
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler

from data_loader import AneurysmSegDataset, TaskListQueue
from utils.project_utils import load_config
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
import torch as th
from guided_diffusion.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist(args)
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    model.to(dist_util.dev())
    
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion,  maxt=600)
    config = load_config('configs/train.yaml')

    logger.log("creating data loader...")

    train_task_list_queue = TaskListQueue(config, 'train', logger, config['data']['train_num_file'], shuffle_files=True)
    train_dataset = AneurysmSegDataset(config, 'train', train_task_list_queue, logger)

    datal= th.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size)
    data = iter(datal)

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        classifier=None,
        data=data,
        dataloader=datal,
        config=config,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="./data/training",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=1000,
        resume_checkpoint='',
        use_fp16=False,
        fp16_scale_growth=1e-3,
        gpu_dev = "1",
        multi_gpu = None #"0,1,2"
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()

