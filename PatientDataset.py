import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader


class PatientDataset(Dataset):
    def __init__(self, root_dir='dataset', image_size=(150, 1050)):
        self.root_dir = root_dir
        self.image_size = image_size
        self.transform_train = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
        self.numeric_features = []
        self.image_features = []
        self.labels = []
        self.gender_encoder = LabelEncoder()
        self.load_data()

    def load_data(self):
        gender_data = []

        # First pass: collect all Gender data for encoding
        for category, patient_path in self._iterate_patient_paths():
            feature_file = os.path.join(patient_path, 'features.csv')
            if os.path.exists(feature_file):
                features = pd.read_csv(feature_file)
                gender_data.extend(features['Gender'].tolist())

        self.gender_encoder.fit(gender_data)

        # Initialize storage structures
        self.numeric_features = []
        self.image_features = []
        self.labels = []

        # Second pass: load data
        for category, patient_path in self._iterate_patient_paths():
            feature_file = os.path.join(patient_path, 'features.csv')
            image_file = os.path.join(patient_path, 'combined_image.jpg')

            if os.path.exists(feature_file) and os.path.exists(image_file):
                # Load features
                features = pd.read_csv(feature_file)
                features['Gender'] = self.gender_encoder.transform(features['Gender'])
                features = features.values.flatten().astype(np.float32)
                self.numeric_features.append(torch.tensor(features, dtype=torch.float32))

                # Load image and ensure it's single channel
                image = Image.open(image_file)
                if image.mode != 'L':
                    image = image.convert('L')

                image = 1.0 - self.transform(image)
                # image = image / 255
                self.image_features.append(image)

                self.labels.append(category)

        # Convert to Tensor
        self.numeric_features = torch.stack(self.numeric_features)
        self.image_features = torch.stack(self.image_features)
        self.labels = LabelEncoder().fit_transform(self.labels)

    # Extract path iteration logic as a helper method to improve readability
    def _iterate_patient_paths(self):
        for category in os.listdir(self.root_dir):
            category_path = os.path.join(self.root_dir, category)
            if os.path.isdir(category_path):
                for patient_id in os.listdir(category_path):
                    patient_path = os.path.join(category_path, patient_id)
                    if os.path.isdir(patient_path):
                        yield category, patient_path

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.image_features[idx], self.numeric_features[idx], self.labels[idx]
