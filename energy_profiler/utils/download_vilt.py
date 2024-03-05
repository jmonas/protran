import torch
import os
from transformers import ViltProcessor, ViltForQuestionAnswering
from datasets import load_dataset
import itertools
from PIL import Image


# Define the path where you want to save the model weights
model_path = '/scratch/gpfs/jmonas/thesis/vilt/model_weights.pth'
data_path = '/scratch/gpfs/jmonas/thesis/vilt/data/vilt_dataset.parquet'

directory = os.path.dirname(model_path)
if not os.path.exists(directory):
    os.makedirs(directory)

directory = os.path.dirname(data_path)
if not os.path.exists(directory):
    os.makedirs(directory)

# Initialize the model with pretrained weights
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa", cache_dir=".cache")
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa", cache_dir=".cache")


# Save the model's state dictionary
# torch.save(model.state_dict(), model_path)

# print(f'Model weights saved to {model_path}')


# Load the VQAv2 dataset
# dataset = load_dataset("HuggingFaceM4/VQAv2", split='train[:5000]')
# print(dataset[0])
# dataset.to_parquet(data_path)
# print(model(dataset[0]))
# # Access the dataset
# print(f'Data saved to {data_path}')



dataset = load_dataset("Graphcore/vqa", split="train[:5000]")
dataset = dataset.remove_columns(['question_type', 'question_id', 'answer_type'])

labels = [item['ids'] for item in dataset['label']]
flattened_labels = list(itertools.chain(*labels))
unique_labels = list(set(flattened_labels))

label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()} 
def replace_ids(inputs):
  inputs["label"]["ids"] = [label2id[x] for x in inputs["label"]["ids"]]
  return inputs


dataset = dataset.map(replace_ids)
flat_dataset = dataset.flatten()

def preprocess_data(examples):
    image_paths = examples['image_id']
    images = [Image.open(image_path) for image_path in image_paths]
    texts = examples['question']    

    encoding = processor(images, texts, padding="max_length", truncation=True, return_tensors="pt")

    for k, v in encoding.items():
          encoding[k] = v.squeeze()
    targets = []

    for labels, scores in zip(examples['label.ids'], examples['label.weights']):
        target = torch.zeros(len(id2label))

        for label, score in zip(labels, scores):
            target[label] = score
        targets.append(target)

    encoding["labels"] = targets
    return encoding

processed_dataset = flat_dataset.map(preprocess_data, batched=True, remove_columns=['question','question_type',  'question_id', 'image_id', 'answer_type', 'label.ids', 'label.weights'])
dataset.to_parquet(processed_dataset)
print(model(processed_dataset[0]))
# Access the dataset
print(f'Data saved to {processed_dataset}')