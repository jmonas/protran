import torch
from torchvision.models import vit_b_16
import os

# Define the path where you want to save the model weights
model_path = '/scratch/gpfs/jmonas/thesis/model_weights.pth'


directory = os.path.dirname(model_path)
if not os.path.exists(directory):
    os.makedirs(directory)


# Initialize the model with pretrained weights
model = vit_b_16(pretrained=True)

# Save the model's state dictionary
torch.save(model.state_dict(), model_path)

print(f'Model weights saved to {model_path}')
