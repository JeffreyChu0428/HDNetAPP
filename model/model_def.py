import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Union, List, Optional
from sklearn.model_selection import train_test_split
from torchhd.embeddings import Projection
from torchhd.models import Centroid

class Patient_Dataset:
    def __init__(self, data_input: list, data_label: Optional[list]=None):
        self.samples = []

        if data_label is not None:
            for x, y in zip(data_input, data_label):
                x_tensor = torch.tensor(x, dtype=torch.float32)
                y_tensor = torch.tensor(y, dtype=torch.long)
                self.samples.append((x_tensor, y_tensor))
        else:
            x_tensor = torch.tensor(data_input, dtype=torch.float32)
            y_tensor = torch.tensor(99, dtype=torch.long)
            self.samples.append((x_tensor, y_tensor))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return x.unsqueeze(0), y  # Add channel dimension: (1, 2500)


class PTBXL_Dataset(Dataset):
    """
    A PyTorch Dataset for ECG samples grouped by subject (PTB-XL).

    Args:
        data_path (str): Path to the .pt file with data grouped by subject.
        subject_ids (Union[float, List[float], None]): Subject(s) to include. If None, uses all subjects.
        split (str): 'train', 'test', or None — whether to return a subset.
        test_ratio (float): Proportion to reserve for test split (if split is specified).
        random_seed (int): Random seed for reproducibility.
    """
    def __init__(
        self,
        data_path: str,
        subject_ids: Optional[Union[float, List[float]]] = None,
        split: Optional[str] = None,
        test_ratio: float = 0.2,
        val_ratio: float = 0.2,
        random_seed: int = 42
    ):
        assert split in [None, 'train', 'val', 'test'], "split must be None, 'train', 'val', or 'test'"

        raw_data = torch.load(data_path)
        all_subject_data = raw_data['data_by_subject']
        self.label_encoder = raw_data['label_encoder']
        # label index: [0:'AFIB' 1:'NORM' 2:'PAC' 3:'PVC' 4:'SBRAD' 5:'STACH']

        # Normalize subject_ids to list
        if subject_ids is None:
            selected_subjects = list(all_subject_data.keys())
        elif isinstance(subject_ids, float):
            selected_subjects = [subject_ids]
        else:
            selected_subjects = subject_ids

        # Collect all (x, y) pairs per subject
        all_samples = []

        for sid in selected_subjects:
            subject_data = all_subject_data[sid]
            x_list = subject_data['x']
            y_list = subject_data['y']
            samples = list(zip(x_list, y_list))
            all_samples.extend(samples)

        # Split based on sample count
        if split is not None:
            stratify_labels = [int(y) for _, y in all_samples]

            # First split into temp (train+val) and test
            temp_idx, test_idx = train_test_split(
                range(len(all_samples)),
                test_size=test_ratio,
                random_state=random_seed,
                stratify=stratify_labels
            )

            temp_samples = [all_samples[i] for i in temp_idx]
            temp_labels = [int(y) for _, y in temp_samples]

            # Now split temp into train and val
            train_idx, val_idx = train_test_split(
                range(len(temp_samples)),
                test_size=val_ratio,
                random_state=random_seed,
                stratify=temp_labels
            )

            if split == 'train':
                indices = [temp_idx[i] for i in train_idx]
            elif split == 'val':
                indices = [temp_idx[i] for i in val_idx]
            elif split == 'test':
                indices = test_idx

            self.samples = [all_samples[i] for i in indices]
        else:
            self.samples = all_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return x.unsqueeze(0), y  # Add channel dimension: (1, 2500)
    
class MITBIH_Dataset(Dataset):
    """
    A PyTorch Dataset for ECG samples grouped by subject (MIT-BIH).

    Args:
        data_path (str): Path to the .pt file with data grouped by subject.
        normal (bool or None): If True, include only normal (label == 0); if False, only abnormal; if None, include all.
        subject_ids (Union[float, List[float], None]): Subject(s) to include. If None, uses all subjects.
        split (str): 'train', 'test', or None — whether to return a subset.
        test_ratio (float): Proportion to reserve for test split.
        random_seed (int): Random seed for reproducibility.
    """
    def __init__(
        self,
        data_path: str,
        normal: Optional[bool] = None,
        subject_ids: Optional[Union[float, List[float]]] = None,
        split: Optional[str] = None,
        test_ratio: float = 0.6,
        random_seed: int = 42
    ):
        assert split in [None, 'train', 'test'], "split must be None, 'train', or 'test'"

        raw_data = torch.load(data_path)
        all_subject_data = raw_data['data_by_subject']
        self.label_encoder = raw_data['label_encoder']

        # Normalize subject_ids
        if subject_ids is None:
            selected_subjects = list(all_subject_data.keys())
        elif isinstance(subject_ids, float):
            selected_subjects = [subject_ids]
        else:
            selected_subjects = subject_ids

        # Collect and filter samples
        all_samples = []
        for sid in selected_subjects:
            subject_data = all_subject_data[sid]
            for x, y in zip(subject_data['x'], subject_data['y']):
                y_int = int(y.item())
                if normal is True and y_int != 0:
                    continue  # keep only label==0
                if normal is False and y_int == 0:
                    continue  # exclude label==0
                if normal is False:
                    y_int -= 1  # shift all labels by -1 so they start from 0

                all_samples.append((x, torch.tensor(y_int)))

        if len(all_samples) == 0:
            raise ValueError("No samples in the dataset. Check filtering conditions.")

        # Optional split
        if split is not None:
            stratify_labels = [int(y) for _, y in all_samples]
            if len(set(stratify_labels)) > 1 and min([stratify_labels.count(cls) for cls in set(stratify_labels)]) >= 2:
                train_idx, test_idx = train_test_split(
                    range(len(all_samples)),
                    test_size=test_ratio,
                    random_state=random_seed,
                    stratify=stratify_labels
                )
            else:
                train_idx, test_idx = train_test_split(
                    range(len(all_samples)),
                    test_size=test_ratio,
                    random_state=random_seed,
                    stratify=None
                )
            indices = train_idx if split == 'train' else test_idx
            self.samples = [all_samples[i] for i in indices]
        else:
            self.samples = all_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return x.unsqueeze(0), y  # Add channel dimension (1, 2500)
    
class ECGNet(nn.Module):
    def __init__(self):
        super(ECGNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 8, 16, stride=2, padding=7),
            nn.ReLU(),
            # nn.BatchNorm1d(8),
            nn.MaxPool1d(kernel_size=8, stride=4),

            nn.Conv1d(8, 12, 12, padding=5, stride=2),
            nn.ReLU(),
            # nn.BatchNorm1d(16),
            nn.MaxPool1d(4, stride=2),

            nn.Conv1d(12, 32, 9, stride=1, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(5, stride=2),

            nn.Conv1d(32, 64, 7, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(4, stride=2),

            nn.Conv1d(64, 64, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),

            nn.Conv1d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),

            nn.Conv1d(64, 72, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
        )
        self.classifier = torch.nn.Sequential(
            nn.Linear(in_features=144, out_features=64),
            nn.ReLU(),
            nn.Dropout(p=.1),
            nn.Linear(in_features=64, out_features=6),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view((x.size(0), -1))
        x = self.classifier(x)
        return x
    
class Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 8, 16, stride=2, padding=7),
            nn.ReLU(),
            # nn.BatchNorm1d(8),
            nn.MaxPool1d(kernel_size=8, stride=4),

            nn.Conv1d(8, 12, 12, padding=5, stride=2),
            nn.ReLU(),
            # nn.BatchNorm1d(16),
            nn.MaxPool1d(4, stride=2),

            nn.Conv1d(12, 32, 9, stride=1, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(5, stride=2),

            nn.Conv1d(32, 64, 7, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(4, stride=2),

            nn.Conv1d(64, 64, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),

            nn.Conv1d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),

            nn.Conv1d(64, 72, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
        )
        self.output = nn.Linear(in_features=144, out_features=64)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.output(x)

class SiameseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = Embedding()
        self.classifier = nn.Sequential(
            nn.Linear(64 * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        f1 = self.embedding(x1)
        f2 = self.embedding(x2)
        distance = torch.nn.functional.pairwise_distance(f1, f2)
        combined = torch.cat([f1, f2], dim=1)
        prob = self.classifier(combined).squeeze()
        return distance, prob
    
class HDNet(nn.Module):
    def __init__(self, normal_database, abnormal_database, normal_extractor_path, abnormal_extractor_path,
                  HD_DIM=10000, FEATURE_DIM=144, NUM_CLASSES=6):
        super().__init__()
        
        self.normal_database = normal_database
        self.abnormal_database = abnormal_database
        self.NUM_CLASSES = NUM_CLASSES
        self.HD_DIM = HD_DIM
        self.FEATURE_DIM = FEATURE_DIM
        self.labels = list(normal_database.label_encoder.classes_) + list(abnormal_database.label_encoder.classes_)

        normal_model = SiameseNet()
        normal_model.load_state_dict(torch.load(normal_extractor_path))
        self.normal_extractor = normal_model.embedding.features

        abnormal_model = ECGNet()
        abnormal_model.load_state_dict(torch.load(abnormal_extractor_path))
        self.abnormal_extractor = abnormal_model.features

        self.normal_projection = Projection(in_features=FEATURE_DIM, out_features=HD_DIM)
        self.abnormal_projection = Projection(in_features=FEATURE_DIM, out_features=HD_DIM)

        self.normal_hv = None
        self.threshold = None
        self.abnormal_classifier = None
        self.ABNORMAL_CLASS_OFFSET = 1

        self.custom_train(self.abnormal_database, init=True)

    def extract_normal_features(self, x):
        with torch.no_grad():
            features = self.normal_extractor(x)         # → shape: (1, C, L)
            return features.view(1, -1)          # → shape: (1, 144)

    def extract_abnormal_features(self, x):
        with torch.no_grad():
            features = self.abnormal_extractor(x)         # → shape: (1, C, L)
            return features.view(1, -1)          # → shape: (1, 144)
        
    def custom_train(self, patient_training_data, threshold=0.95, init=False):
        all_hvs=[]

        if init == False:
            patient_hvs = []   
            for idx in range(patient_training_data.__len__()):
                x, y = patient_training_data.__getitem__(idx)
                x = x.unsqueeze(0)  # (1, 1, 2500)
                if y.item() == 0:
                    feat = self.extract_normal_features(x)     # → shape: (1, feature_dim)
                    hv = self.normal_projection(feat)                 # → shape: (1, hd_dim)
                    hv = torch.nn.functional.normalize(hv, dim=1)  # normalize to unit vector
                    patient_hvs.append(hv.squeeze(0))         # shape: (hd_dim,)
            if len(patient_hvs) > 0:
                patient_hv = torch.stack(patient_hvs).mean(dim=0)
                all_hvs.append(patient_hv)

        if len(all_hvs) == 0:
            base_hvs = []
            for idx in range(10):
                x, y = self.normal_database.__getitem__(idx)
                x = x.unsqueeze(0)  # (1, 1, 2500)
                if y.item() == 0:
                    feat = self.extract_normal_features(x)     # → shape: (1, feature_dim)
                    hv = self.normal_projection(feat)                 # → shape: (1, hd_dim)
                    hv = torch.nn.functional.normalize(hv, dim=1)  # normalize to unit vector
                    base_hvs.append(hv.squeeze(0))         # shape: (hd_dim,)
        
            base_hv = torch.stack(base_hvs, dim=0).mean(dim=0)  # → shape: (hd_dim,)
            all_hvs.append(base_hv)

        normal_hv = torch.stack(all_hvs).mean(dim=0)  # centroid
        self.normal_hv = torch.nn.functional.normalize(normal_hv, dim=0)  # final normalized prototype
        self.threshold = threshold

        self.abnormal_classifier = Centroid(self.HD_DIM, self.NUM_CLASSES)

        for idx in range(self.abnormal_database.__len__()):
            x, y = self.abnormal_database.__getitem__(idx)
            x = x.unsqueeze(0)  # shape: (1, 1, 2500)
            y = y.unsqueeze(0)
            feat = self.extract_abnormal_features(x)     # → shape: (1, 144)
            hv = self.abnormal_projection(feat)          # → shape: (1, 10000)
            self.abnormal_classifier.add(hv, y)       # add to HDC class prototype

        if init == False:
            for idx in range(patient_training_data.__len__()):
                x, y = patient_training_data.__getitem__(idx)
                if y.item() != 0:
                    x = x.unsqueeze(0)  # (1, 1, 2500)
                    y = y.unsqueeze(0)
                    feat = self.extract_abnormal_features(x)     # → shape: (1, 144)
                    hv = self.abnormal_projection(feat)          # → shape: (1, 10000)
                    self.abnormal_classifier.add(hv, y)       # add to HDC class prototype

        # Normalize after training
        self.abnormal_classifier.normalize()

    def test(self, patient_testing_data):
        pred_labels=[]
        normal_similarities = []

        for idx in range(patient_testing_data.__len__()):
            x, y = patient_testing_data.__getitem__(idx)
            x = x.unsqueeze(0)  # (1, 1, 2500)
            y = y.item()

            # Step 1: Extract features & encode to HD space
            feat = self.extract_normal_features(x)
            hv = self.normal_projection(feat)
            hv = torch.nn.functional.normalize(hv, dim=1)
            hv = hv.squeeze(0)  # shape: (hd_dim,)

            # Step 2: Cosine similarity to prototype
            similarity = torch.nn.functional.cosine_similarity(hv, self.normal_hv, dim=0).item()
            normal_similarities.append(similarity)

            # Step 3: Classification decision
            if similarity >= self.threshold:
                pred_labels.append(0)# 0 = normal
            else:
                feats = self.extract_abnormal_features(x)   # shape: (1, 144)
                hv = self.abnormal_projection(feats)              # shape: (1, 10000)
                out = self.abnormal_classifier(hv)             # shape: (1, num_classes)
                with torch.no_grad():
                    pred = out.argmax(dim=1).item()
                pred_labels.append(pred+self.ABNORMAL_CLASS_OFFSET)

        return pred_labels, normal_similarities
        