import argparse
import os

import sys
import random
sys.path.append(".")
import numpy as np
import torch as th

from data_loader import AneurysmSegTestManager
from inference import Inferencer
from utils.project_utils import load_config
from collections import OrderedDict

from guided_diffusion import dist_util, logger

from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

seed=10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

def dice_score(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist(args)
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    config = load_config('configs/inference.yaml')

    logger.log("creating data loader...")
    
    devices = dist_util.dev()

    inference_data_manager = AneurysmSegTestManager(config, logger, devices)

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    
    input_file_or_folder = args.data_dir
    output_folder = 'Path to the file where the prediction results are saved'
    input_type = 'nii'
    save_binary = 'true'
    print(devices)

    inferencer = Inferencer(config, model, diffusion, args.num_ensemble, args.use_ddim, args.image_size, args.batch_size, args.clip_denoised,
                            input_file_or_folder, output_folder, input_type, save_binary, inference_data_manager)
    inferencer.inference()
  

def create_argparser():
    defaults = dict(
        data_dir="input image path",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="",
        num_ensemble=5,
        gpu_dev = "1",
        out_dir='./results/',  #"0,1,2"
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":

    main()
