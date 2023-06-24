import os
import json
from torch.utils.data.dataset import IterableDataset
from datasets import Dataset
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate


class SimpleDataSet(IterableDataset):

    def __init__(self, path):
        self.path = path
        self.samples = []
        self.load()

    def load(self):
        projects = os.listdir(self.path)
        for project in projects:
            project_path = os.path.join(self.path, project)
            json_files = os.listdir(project_path)
            for json_file in json_files:
                with open(os.path.join(project_path, json_file), 'r') as file:
                    json_data = json.load(file)
                    files = json_data['files']
                    for file in files:
                        for t in file['positives']:
                            sample = {}
                            sample['text'] = t['code']
                            sample['label'] = 1
                            self.samples.append(sample)
                        for t in file['negatives']:
                            sample = {}
                            sample['text'] = t['code']
                            sample['label'] = 0
                            self.samples.append(sample)

    def __iter__(self):
        for data in self.samples:
            yield data

    def __len__(self):
        return len(self.samples)

    def get_hf_dataset(self):
        return Dataset.from_generator(self.__iter__)



path = "/zfy/apache_jit/output"
model_path = "/zfy/models/model.pt"

dataset = SimpleDataSet(path)

hfds = dataset.get_hf_dataset()

from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)


def tokenize_function(sample):
    return tokenizer(sample['text'], padding="max_length", truncation=True)

tokenized_datasets = hfds.map(tokenize_function, batched=True)


ds = tokenized_datasets.train_test_split(test_size=0.2, shuffle=True)
print(ds)

training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy='epoch')

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds['train'],
    eval_dataset=ds['test'],
    compute_metrics=compute_metrics
)

trainer.train()