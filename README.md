# MFDiff-master
MFDiff
# Requirements

The project has been tested in Ubuntu 20.04 using python 3.7.
* torch
* torchvision
* tensorboardX
* medpy
* SimpleITK
* PyYAML
* pillow
* colorlog

1. Data

Prepare an data information file like `medical_data/aneurysm_seg_z.csv`. All the images should be in nii.gz format.

This folder stores the training and evaluation dataset. You should modify dataset information in aneurysm_seg_z.csv.  And the cta_img_file and aneurysm_seg_file are the paths to the image and the lab image, respectively.

2. Train

If you want to train your model, run: ``python guided_scripts/segmentation_train.py --data_dir ./data/training  --image_size 128 --num_channels 8 --class_cond False --num_res_blocks 1 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions "32" --diffusion_steps 600 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --batch_size 2``

The model will be saved in the 'results' folder.

3. eval

If you want to evaluate segmentation results, run: ``python guided_scripts/segmentation_sample.py  --model_path .pt file path --num_ensemble=1 --image_size 128 --num_channels 8 --class_cond False --num_res_blocks 1 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions "32" --diffusion_steps 600 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False``

4. inference

If you want to generate predictions, i.e. .nii.gz images, run: ``python guided_scripts/segmentation_inference.py  --data_dir .nii data or folder path --model_path .pt file path  --num_ensemble=1 --image_size 128 --num_channels 8 --class_cond False --num_res_blocks 1 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions "32" --diffusion_steps 30 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False``
