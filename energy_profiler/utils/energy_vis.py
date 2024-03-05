import os
import sys

sys.path.append('../tests/')
sys.path.append('../../txf_design-space/transformers/src/transformers')

import json
import time
import shlex
import shutil
import platform
import subprocess
import numpy as np
from pathlib import Path
import multiprocessing as mp
from matplotlib import pyplot as plt
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import vit_b_16
from torch.utils.data import DataLoader

# from run_glue import main as run_glue
# # try:
# from run_glue_onnx import main as run_glue_onnx
# # except:
    # print('Could not import ONNX library')


from transformers.convert_graph_to_onnx import convert
from datasets import load_dataset



def get_power(device: str = 'gpu', debug: bool = True):
    """Get current power consumption
    
    Args:
        device (str, optional): device in ['cpu', 'gpu', 'npu']
        rpi_ip (str, optional): IP address of the RPi connected to the INA219 sensor for power measurement
        debug (bool, optional): print statements if True

    Raises:
        RunTimeError: if OS is not supported

    Returns:
        dict: power metrics in mW
    """
    cpu_power, gpu_power, dram_power = 0, 0, 0
    print("Get Power")
    if device == 'gpu':
        # Get raw output of nvidia-smi
        power_stdout = subprocess.check_output(
            f'nvidia-smi --query --display=POWER --id=0', # Assuming GPU-id to be 0 for now
            shell=True, text=True)
        
        print(power_stdout)
        power_stdout = power_stdout.split('\n')
        
        for line in power_stdout:
            if 'Draw' in line.split():
                if line.split()[-2] != 'N/A':
                    try:
                        float(line.split()[-2])
                        gpu_power = float(line.split()[-2])
                    except:
                        continue

        if debug: print(f'GPU Power: {gpu_power : 0.02f} W')
        print('gpu:', gpu_power * 1000)
        return {'gpu': gpu_power * 1000}
    


def run_inference(queue, device: str, batch_size: int, runs: int, model_path: str):
    """Run inference of given model on the given GLUE task
    
    Args:
        queue (mp.Queue): multiprocessing queue
        device (str): device in ['cpu', 'gpu', 'npu']
        max_seq_length: maximum sequence length
        batch_size (int): batch size to be used for running inference
        runs (int): number of inference runs
        model_path (str): directory where the model is stored
        task (str): GLUE task to run inference on
        num_samples (int, optional): number of samples in the validation set to run partial inference
    
    Returns:
        dict: evaluation metrics
    """
    print("INFERENCE BLOCK----")

    # Define the image transformations for preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),  # Resize the image to 256x256 pixels
        transforms.CenterCrop(224),  # Crop the image to 224x224 pixels about the center
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize pixel values (ImageNet mean and std)
    ])
    print("Download imagenet----")

    # Load the ImageNet dataset from the local directory with transformations applied
    imagenet_dataset = datasets.ImageNet('/scratch/gpfs/DATASETS/imagenet/ilsvrc_2012_classification_localization', transform=preprocess)

    # Create a DataLoader to handle batching and shuffling
    data_loader = DataLoader(imagenet_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Initialize the model without pretrained weights
    model = vit_b_16(pretrained=False)


    # Load the weights into the model
    model.load_state_dict(torch.load(model_path, map_location='cuda'))

    # Move the model to CUDA
    model = model.to('cuda')
    model.eval()  # Set the model to evaluation mode

    print("warm up----")

    # Warm-up
    for _ in range(50):
        inputs = torch.randn(1, 3, 224, 224).to('cuda')
        with torch.no_grad():
            model(inputs)

    print("begin main inference----")

    start_time = time.time()
    for i in range(runs):
        print("run: ",i)
        if device == 'gpu':
            print("i am a gpu")
            # We assume only one GPU is avilable. Else, use: os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            with torch.no_grad():
                for images, _ in data_loader:
                    images = images.to('cuda')
                    print("len of images: ", len(images))
                    output = model(images)
                    print("OUTPUT:")
                    print(output)
            
    end_time = time.time()
    eval_metrics = {'eval_loss': np.nan, 'eval_accuracy': np.nan, 'eval_runtime': end_time - start_time}
    # eval_metrics = json.load(open(os.path.join(model_path, 'eval_results.json'), 'r'))

    queue.put(eval_metrics)



def _find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def get_measures(device: str, 
    model_path: str, 
    batch_size: int, 
    runs: int, 
    debug: bool = True):
    """Get hardware performance measures - latency, energy, and peak power consumption per run of inference on the given task
    
    Args:
        device (str): device in ['cpu', 'gpu', 'npu']
        model_path (str): directory where the model is stored
        batch_size (int): batch size to be used for running inference
        max_seq_length (int): maximum sequence length for running inference
        runs (int): number of inference runs
        debug (bool, optional): to pring debug statements and save power consumption figures
    
    Raises:
        RuntimeError: if Intel NCS is to be used, but root user is not selected
    """

    print("start job---")
    # Get mutliprocessing queue
    vit_queue = mp.Queue()

    # Get process
    vit_process = mp.Process(target=run_inference, args=(vit_queue, device, batch_size, runs, model_path))

    start_time = time.time()
    power_metrics = []

    print("Get power consumption for first 5 iterations")

    # Get power consumption for first 5 iterations
    print("hi")
    for i in range(5):
        power_metrics.append({'power_metrics': get_power(device=device, debug=debug), 'time': time.time() - start_time})
        if platform.system() == 'Linux': 
            time.sleep(0.2)

    # Start inference of of the given model for 'runs' runs
    print("Start Process")
    vit_process.start()


    iterations = 500

    # Initialize evaluation runtime variables
    eval_start_time = 0
    eval_runtime = 0

    print("Get power consumption for more iterations")

    # Get power consumption for more iterations
    start_counter, counter = False, 5
    for i in range(iterations):
        if start_counter: counter -= 1 # Inference ended, run for 5 more iterations
        if counter == 0: break
        power_metrics.append({'power_metrics': get_power(device=device, debug=True), 'time': time.time() - start_time})
        if platform.system() == 'Linux': 
            time.sleep(0.2)
        if vit_process.is_alive() and eval_start_time == 0:
            eval_start_time = time.time() - start_time
        if not vit_process.is_alive() and eval_runtime == 0:
            eval_runtime = time.time() - eval_start_time - start_time
            start_counter = True

    # Get metrics from common queue
    eval_metrics = vit_queue.get()

    # Join process
    vit_process.join()

    # Update evaluation metrics with better runtime estimate
    eval_metrics['eval_runtime'] = eval_runtime

    # Fix timing
    exp_start_time = power_metrics[0]['time']
    for i in range(len(power_metrics)):
        power_metrics[i]['time'] -= exp_start_time

    # Find number of sequences in the dataset
    # if task != 'glue':
    #     num_sequences = load_dataset('glue', task, split='validation').num_rows
    # else:
    #     num_sequences = 0
    #     for task in GLUE_TASKS:
    #         num_sequences += load_dataset('glue', task, split='validation').num_rows

    num_sequences = 1

    print("Graph")
    if platform.system() == 'Linux':
        color = 'b'
        if device == 'npu':
            color = 'tab:orange'
        else:
            color = 'g'

        # Get energy
        _, eval_start_idx = _find_nearest([meas['time'] for meas in power_metrics], eval_start_time)
        _, eval_end_idx = _find_nearest([meas['time'] for meas in power_metrics], eval_start_time+eval_metrics['eval_runtime'])
        energy = np.trapz([meas['power_metrics'][device]/1000 for meas in power_metrics][eval_start_idx:eval_end_idx], 
            [meas['time'] for meas in power_metrics][eval_start_idx:eval_end_idx])

        # Get peak power in W
        peak_power = max([meas['power_metrics'][device] for meas in power_metrics][eval_start_idx:eval_end_idx])/1000 

        # Make plot
        fig, ax1 = plt.subplots(1, 1)
        ax1.plot([meas['time'] for meas in power_metrics], [meas['power_metrics'][device] for meas in power_metrics], label=f'{device.upper()} Power', 
            color=color)

        ax1.axvline(x=eval_start_time, linestyle='--', color='k')
        ax1.axvline(x=eval_start_time+eval_metrics['eval_runtime'], linestyle='--', color='k')

        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel(f'{device.upper()} Power (mW)')

        # ax1.set_title(f'Model: BERT-Tiny | Task: {task} | No. sequences: {num_sequences} \n Energy: {energy/runs : 0.2f}J/run | Runtime: {eval_metrics["eval_runtime"]/runs : 0.2f}s/run')
    print("Dump results")
    if debug: 
        plt.savefig(os.path.join(model_path, 'power_results.pdf'), bbox_inches='tight')
        print(f'Evaluation Accuracy (%): {eval_metrics["eval_accuracy"]*100}. Evaluation Runtime (s/run): {eval_metrics["eval_runtime"]/runs}')

    json.dump(power_metrics, open(os.path.join(model_path, 'power_metrics.json'), 'w+'))

    protran_results = {'latency': eval_metrics["eval_runtime"]/runs/num_sequences, 'energy': energy/runs/num_sequences, 'peak_power': peak_power}
    json.dump(protran_results, open(os.path.join(model_path, 'protran_results.json'), 'w+'))

    return protran_results










get_measures("gpu","/scratch/gpfs/jmonas/thesis/model_weights.pth", 32, 10)