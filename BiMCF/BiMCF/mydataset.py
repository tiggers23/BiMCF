import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, ViTModel
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from PIL import Image
from transformers import BertTokenizer, ViTFeatureExtractor


# ################ DATASET AND DATALOADER ######################
class MyDataset(Dataset):
    def __init__(self, file_path, tokenizer_name='bert-base-uncased', feature_extractor_name='google/vit-base-patch16-224-in21k'):
        self.data = []
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(feature_extractor_name)
        
        with open(file_path, 'r') as f:
            for line in f:
                image_path, text, task1_label, task2_label, task3_label = line.strip().split('\t')
                self.data.append((image_path, text, int(task1_label), int(task2_label), int(task3_label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, text, task1_label, task2_label, task3_label = self.data[idx]
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        pixel_values = self.feature_extractor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        
        # Tokenize text
        encoding = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'task1_label': torch.tensor(task1_label, dtype=torch.long),
            'task2_label': torch.tensor(task2_label, dtype=torch.long),
            'task3_label': torch.tensor(task3_label, dtype=torch.long)
        }
