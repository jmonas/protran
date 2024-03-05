import torch
import os
from transformers import ViltProcessor, ViltForQuestionAnswering
from datasets import load_dataset



# Define the path where you want to save the model weights
model_path = '/scratch/gpfs/jmonas/thesis/vilt/model_weights.pth'
data_path = '/scratch/gpfs/jmonas/thesis/vilt/data/'

directory = os.path.dirname(model_path)
if not os.path.exists(directory):
    os.makedirs(directory)

directory = os.path.dirname(data_path)
if not os.path.exists(directory):
    os.makedirs(directory)

# Initialize the model with pretrained weights
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# Save the model's state dictionary
torch.save(model.state_dict(), model_path)

print(f'Model weights saved to {model_path}')


# Load the VQAv2 dataset
dataset = load_dataset("HuggingFaceM4/VQAv2", split='train')
dataset.to_parquet(data_path)
# Access the dataset
print(f'Data saved to {data_path}')
