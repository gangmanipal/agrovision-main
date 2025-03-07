import pickle
import pandas as pd
import numpy as np
import time
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
from imblearn.over_sampling import SMOTE

# Load dataset
file_path = "Data/Crop_recommendation.csv"
df = pd.read_csv(file_path)
X = df.drop(columns=['label'])
y = df['label']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Apply SMOTE for class balancing
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Wrapper for TabNet to work with Scikit-Learn
class SklearnTabNetClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.model = TabNetClassifier(optimizer_params=dict(lr=8e-3), verbose=0)

    def fit(self, X, y):
        X = X.to_numpy()
        y = np.array(y)
        self.model.fit(X, y, eval_set=[(X, y)], patience=50, max_epochs=400)
        return self

    def predict(self, X):
        X = X.to_numpy()
        return self.model.predict(X)

    def predict_proba(self, X):
        X = X.to_numpy()
        return self.model.predict_proba(X)

# Define base models for stacking
base_learners = [
    ('xgb', XGBClassifier(n_estimators=300, max_depth=10, learning_rate=0.01, subsample=0.8, eval_metric='mlogloss', random_state=42)),
    ('catboost', CatBoostClassifier(iterations=300, depth=10, learning_rate=0.01, random_seed=42, verbose=0)),
    ('tabnet', SklearnTabNetClassifier())
]

# Define meta-model (Neural Network MLP)
meta_model = MLPClassifier(hidden_layer_sizes=(1024, 512, 256, 128), activation='relu', solver='adam', max_iter=700, batch_size=128, random_state=42, early_stopping=True)

# Define stacking classifier with passthrough for weighted influence
stacked_model = StackingClassifier(estimators=base_learners, final_estimator=meta_model, cv=5, passthrough=True)

if __name__ == "__main__":
    start_time = time.time()
    
    # Train stacked model
    stacked_model.fit(X_train, y_train)
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # Predictions
    y_pred = stacked_model.predict(X_test)

    # Calculate accuracy
    stacked_accuracy = accuracy_score(y_test, y_pred)
    print(f"High-Accuracy Stacked Model Accuracy: {stacked_accuracy * 100:.2f}%")
    print(f"Training Time: {training_time:.2f} seconds")

    # Save models separately
    with open('models/StackedModel.pkl', 'wb') as model_file:
        pickle.dump(stacked_model, model_file)
    with open('models/LabelEncoder.pkl', 'wb') as encoder_file:
        pickle.dump(label_encoder, encoder_file)