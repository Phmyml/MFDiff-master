import argparse

import sys
import random
sys.path.append(".")
import numpy as np
import torch as th


from data_loader import AneurysmSegDataset, TaskListQueue
from utils.project_utils import load_config
from collections import OrderedDict

from guided_diffusion import dist_util, logger

from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from utils.metrics import get_evaluation_metric

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

def _build_batch_and_normalize(inputs, config, targets=None):
        # all as dict
        model_inputs = OrderedDict()
        if targets is not None:
            model_targets = OrderedDict()  # the first is main
        raw_inputs = OrderedDict()  # for visualization
        hu_intervals = config['data']['hu_values']

        def _normalize(image):
            normalized_img = []
            for hu_inter in hu_intervals:
                hu_channel = th.clamp(image, hu_inter[0], hu_inter[1])
                # norm to 0-1
                normalized_img.append((hu_channel - hu_inter[0]) / (hu_inter[1] - hu_inter[0]))
            normalized_img = th.stack(normalized_img, dim=1)
            return normalized_img

        if config['task'] == 'AneurysmSeg':
            model_inputs['local_cta_input'] = _normalize(inputs['cta_img']).type(th.float32)
            raw_inputs['local_cta_input'] = th.unsqueeze(inputs['cta_img'].type(th.float32), 1)
            if targets is not None:
                model_targets['local_ane_seg_target'] = targets['aneurysm_seg'].type(th.int64)
        else:
            logger.log('Cannot recognize task: %s' % config['task'])
            exit(1)
        return model_inputs, model_targets

def _reset_eval_metrics(eval_metric_fns):
    for metric_fn in eval_metric_fns.values():
        metric_fn.reset()

def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist(args)
    logger.configure()
    print(args.diffusion_steps)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    config = load_config('configs/eval.yaml')

    logger.log("creating data loader...")
    
    sample_task_list_queue = TaskListQueue(config, 'eval', logger, config['data']['eval_num_file'], shuffle_files=True)
    sample_dataset = AneurysmSegDataset(config, 'eval', sample_task_list_queue, logger)
    datal= th.utils.data.DataLoader(
        sample_dataset,
        batch_size=args.batch_size)
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    phase='eval'
    
    eval_avg_losses = None
    eval_metric_fns, eval_curve_fns = get_evaluation_metric(config, logger)
    _reset_eval_metrics(eval_metric_fns)   

    time_per_iter = None
    log_every_n_iters = 1
    prob_threshold = 0.5

    with th.no_grad():
        for i, data in enumerate(datal):
            inputs, targets, metas = data
            inputs, targets= _build_batch_and_normalize(inputs, config, targets)
            b = inputs['local_cta_input']
            target = targets['local_ane_seg_target'].to(dist_util.dev())

            c = th.randn_like(b[:, :1, ...])
            img = th.cat((b, c), dim=1)
            start = th.cuda.Event(enable_timing=True)
            end = th.cuda.Event(enable_timing=True)

            for j in range(args.num_ensemble):  #this is for the generation of an ensemble of 5 masks.
                model_kwargs = {}
                start.record()
                sample_fn = (
                    diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
                )
                sample_f, x_noisy, org, cal, cal_out, cal_out_softmax = sample_fn(
                    model,
                    (args.batch_size, 4, 96, 96), img,step=args.diffusion_steps,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                )
                
                sample = th.where(cal_out > prob_threshold, 1, 0)
                current_metrics = OrderedDict()
                
                for key, metric_fn in eval_metric_fns.items():                   
                    current_metrics[key] = metric_fn(sample, target, **metas)         
                logging_info = 'one batch_size finished.'

                if (i + 1) % log_every_n_iters == 0:
                    for metric_name, metric_value in current_metrics.items():
                        if isinstance(metric_value.item(), int):
                            logging_info += ' %s: %d' % (metric_name, metric_value.item())
                        else:
                            logging_info += ' %s: %1.4f' % (metric_name, metric_value.item())
                    logger.log(logging_info)

                end.record()
                th.cuda.synchronize()
                print('time for 1 sample', start.elapsed_time(end))  #time measurement for the generation of 1 sample
        logging_info = 'eval finished.'
        for metric_name, metric_fn in eval_metric_fns.items():
            if isinstance(metric_fn.result.item(), int):
                logging_info += ' %s: %d' % (metric_name, metric_fn.result.item())
            else:
                logging_info += ' %s: %1.4f' % (metric_name, metric_fn.result.item())
        logger.log(logging_info)


def create_argparser():
    defaults = dict(
        data_dir="./medical_data/img_crop/L1",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="",
        num_ensemble=5,      # number of samples in the ensemble1
        gpu_dev = "1",
        out_dir='./results/',
        multi_gpu = None  #"0,1,2"
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":

    main()
