import os
import time
from collections import OrderedDict

import numpy as np
import torch
from tensorboardX import SummaryWriter

from data_loader import get_instances_from_file_or_folder
from utils.project_utils import maybe_create_path

from guided_diffusion import logger

class _ModelCore:
    def __init__(self,
                 config,
                 train_loader=None,
                 eval_loader=None,
                 test_loader=None,
                ):
        self.config = config
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.test_loader = test_loader

        self.num_iterations = 1
        self.num_epoch = 1
        self.best_eval_score = None

    
    def _build_batch_and_normalize(self, inputs, targets=None):
        model_inputs = OrderedDict()
        if targets is not None:
            model_targets = OrderedDict()
        raw_inputs = OrderedDict()
        hu_intervals = self.config['data']['hu_values']

        def _normalize(image):
            normalized_img = []
            for hu_inter in hu_intervals:
                hu_channel = torch.clamp(image, hu_inter[0], hu_inter[1])   
                # norm to 0-1
                normalized_img.append((hu_channel - hu_inter[0]) / (hu_inter[1] - hu_inter[0]))
            normalized_img = torch.stack(normalized_img, dim=1)
            return normalized_img

        if self.config['task'] == 'AneurysmSeg':
            model_inputs['local_cta_input'] = _normalize(inputs['cta_img']).type(torch.float32)
            raw_inputs['local_cta_input'] = torch.unsqueeze(inputs['cta_img'].type(torch.float32), 1)
            if targets is not None:
                model_targets['local_ane_seg_target'] = targets['aneurysm_seg'].type(torch.int64)
                
        else:
            self.logger.critical('Cannot recognize task: %s' % self.config['task'])
            exit(1)

        if targets is not None:
            
            return model_inputs, model_targets 
        else:
            return model_inputs

    def _compute_loss_and_output(self, raw_outputs, targets=None, weights=None):

        # apply final activation to outputs
        outputs = OrderedDict()
        for j, key in enumerate(raw_outputs.keys()):
            if self.config['train']['losses'][j]['final_activation'] == 'softmax':
                outputs[key] = torch.nn.Softmax(dim=1)(raw_outputs[key])
            elif self.config['train']['losses'][j]['final_activation'] == 'sigmoid':
                fg_output = torch.nn.Sigmoid()(raw_outputs[key])
                outputs[key] = torch.cat([1 - fg_output, fg_output], dim=1)
            elif self.config['train']['losses'][j]['final_activation'] == 'identity':
                outputs[key] = raw_outputs[key]
        # outputs and losses have same length

        if targets is None:
            return outputs
        else:
            return outputs

    

class Inferencer(_ModelCore):
    def __init__(self,
                 config,
                 model,
                 diffusion,
                 num_ensemble,
                 use_ddim,
                 image_size,
                 batch_size,
                 clip_denoised,
                 inference_file_or_folder,
                 output_folder=None,
                 input_type='nii',
                 save_binary=True,
                 test_loader_manager=None,):
        super(Inferencer, self).__init__(config, test_loader=test_loader_manager.test_loader)
        self.test_loader_manager = test_loader_manager
        self.test_phase = config['eval'].get('phase', 'inference')
        if input_type not in ['nii', 'dcm']:
            logger.log('input_type must be nii or dcm')
            exit(1)
        if not os.path.exists(inference_file_or_folder):
            logger.log('inference_file_or_folder %s does not exist.' % inference_file_or_folder)
            exit(1)
        self.inference_file_or_folder = inference_file_or_folder
        self.model = model
        self.diffusion = diffusion
        self.num_ensemble = num_ensemble
        self.use_ddim = use_ddim
        self.image_size = image_size
        self.batch_size = batch_size
        self.clip_denoised = clip_denoised
        self.output_folder = output_folder
        self.input_type = input_type
        self.save_binary = save_binary
        
        if 'eval' in config:
            self.prob_threshold = config['eval'].get('probability_threshold', 0.5)
        else:
            self.prob_threshold = 0.3


    def inference(self):
        if self.test_loader is None:
            logger.log('Try to %s but there is no test_loader' % self.test_phase)
            return 0.0

        logger.info('Begin to scan input_folder_or_file %s...' % self.inference_file_or_folder)
        instances = get_instances_from_file_or_folder(self.inference_file_or_folder, instance_type=self.input_type)
        for i, instance in enumerate(instances):
            if self.output_folder is None:
                if self.input_type == 'nii':
                    output_file = instance.replace('.nii.gz', '_pred.nii.gz')
                elif self.input_type == 'dcm':
                    output_file = os.path.join(os.path.dirname(instance[0]), 'prediction.nii.gz')
            else:
                assert not instance[0].startswith(self.output_folder)  # in case override original
                if self.input_type == 'nii':
                    output_file = os.path.join(self.output_folder, os.path.basename(instance))
                elif self.input_type == 'dcm':
                    output_file = os.path.join(self.output_folder,
                                               os.path.basename(os.path.dirname(instance[0])) + '.nii.gz')
            self.inference_instance(instance, output_file)
            logger.log('finish %d in %d instances' % (i + 1, len(instances)))

    def inference_instance(self, input_file_s, output_file):
        self.test_loader_manager.load(input_file_s, input_type=self.input_type)
        self.model.eval()

        instance_start_time = time.time()
        prediction_instance_shape = self.test_loader_manager.instance_shape
        prediction_patch_starts = self.test_loader_manager.patch_starts
        prediction_patch_size = self.test_loader_manager.patch_size

        prediction = np.zeros(prediction_instance_shape.tolist(), dtype=np.float32)
        
        overlap_count = np.zeros(prediction_instance_shape.tolist(), dtype=np.float32)

        if isinstance(input_file_s, list) or isinstance(input_file_s, tuple):
            input_instance = os.path.dirname(input_file_s[0])
        else:
            input_instance = input_file_s

        preprocess_time = time.time() - instance_start_time

        logger.log('%s instance %s (%d patches)...'
                         % (self.test_phase, input_instance, len(prediction_patch_starts)))
        time_all_iters = 0
        time_loading = 0

        print('\r\tprocessing procedure: %1.1f%%' % 0, end='')
        with torch.no_grad():
            loading_since = time.time()
            for i, data in enumerate(self.test_loader):
                time_loading += time.time() - loading_since
                since = time.time()
                inputs, metas = data
                
                inputs= self._build_batch_and_normalize(inputs)
                b = inputs['local_cta_input']     
                c = torch.randn_like(b[:, :1, ...])
                img = torch.cat((b, c), dim=1)

                logger.log("sampling...")

                for z in range(self.num_ensemble):
                    model_kwargs = {}
                    sample_fn = (
                        self.diffusion.p_sample_loop_known if not self.use_ddim else self.diffusion.ddim_sample_loop_known
                    )
                    sample_f, x_noisy, org, cal, cal_out, cal_out_softmax = sample_fn(
                        self.model,
                        (self.batch_size, 4, 96, 96), img,step=600,
                        clip_denoised=self.clip_denoised,
                        model_kwargs=model_kwargs,
                    )
                    cal_out_1 = cal_out[:,1]
                    sample = cal_out_1.detach().cpu().numpy()



                # iter along batch
                for j, main_out in enumerate(sample):
                    patch_starts = metas['patch_starts'][j]
                    patch_ends = [patch_starts[i] + prediction_patch_size[i] for i in range(3)]

                    prediction[patch_starts[0]:patch_ends[0], patch_starts[1]:patch_ends[1],
                    patch_starts[2]:patch_ends[2]] += main_out
                    overlap_count[patch_starts[0]:patch_ends[0], patch_starts[1]:patch_ends[1],
                    patch_starts[2]:patch_ends[2]] += 1

                time_all_iters += time.time() - since
                print('\r\tprocessing procedure: %1.1f%%' % (
                            100 * self.batch_size * (i + 1) / len(prediction_patch_starts)), end='')
                loading_since = time.time()

            print('')
            if time_all_iters == 0:
                self.logger.error('\tNo %s data fetched from instance %s. Try to restart...'
                                % (self.test_phase, input_instance))
                return self.inference_instance(input_file_s, output_file)

            summarize_since = time.time()
            print('\tsummarize and save...')
            overlap_count = np.where(overlap_count == 0, np.ones_like(overlap_count), overlap_count)
            prediction = prediction / overlap_count
            prediction = self.test_loader_manager.restore_spacing(prediction, is_mask=False)

            summarize_time = time.time() - summarize_since
            processing_time = time.time() - instance_start_time

            if self.output_folder is None:
                assert '_pred' in output_file  # in case override original
            else:
                maybe_create_path(self.output_folder)
            if self.save_binary:
                binary_prediction = (prediction > self.prob_threshold).astype(np.int32)
                self.test_loader_manager.save_prediction(binary_prediction, output_file)
            

            logging_info = '\t((total time: %1.2f; preprocess_time: %1.2f; loading_time: %1.2f; network time: %1.2f; summarize_time: %1.2f) Prediction of instance %s saved to %s. ' \
                           % (processing_time, preprocess_time, time_loading, time_all_iters, summarize_time,
                               input_instance, output_file)
            print(logging_info)
            return prediction

def _build_batch_and_normalize(inputs):
        
        hu_intervals = [-100,1000]

        normalized_img = []
        inputs = torch.from_numpy(inputs)
        hu_channel = torch.clamp(inputs, -100, 1000)
        # norm to 0-1
        hu_channel = hu_channel.numpy()
        
        normalized_img.append((hu_channel - hu_intervals[0]) / (hu_intervals[1] - hu_intervals[0]))
        normalized_img = torch.stack(normalized_img, dim=1)

        return normalized_img
