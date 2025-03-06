import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd


def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                        nn.Flatten(),
                                        nn.Linear(512, num_diseases))

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out


# Load dataset for training the stacked model
file_path = "Data/Crop_recommendation.csv"
df = pd.read_csv(file_path)
X = df.drop(columns=['label'])
y = df['label']

# Standardize the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode the labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data for evaluation
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Define base models
base_learners = [
    ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)),
    ('lgbm', LGBMClassifier(random_state=42))
]

# Define meta-model
meta_model = LogisticRegression()

# Define stacking classifier
stacked_model = StackingClassifier(estimators=base_learners, final_estimator=meta_model, cv=5)

if __name__ == "__main__":
    # Train stacked model
    stacked_model.fit(X_train, y_train)

    # Predictions
    y_pred = stacked_model.predict(X_test)

    # Calculate accuracy
    stacked_accuracy = accuracy_score(y_test, y_pred)
    print(f"Optimized Stacked Model Accuracy: {stacked_accuracy * 100:.2f}%")

    # Save models separately
    with open('models/StackedModel.pkl', 'wb') as model_file:
        pickle.dump(stacked_model, model_file)
    with open('models/Scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
    with open('models/LabelEncoder.pkl', 'wb') as encoder_file:
        pickle.dump(label_encoder, encoder_file)