import pickle
import pandas as pd
import numpy as np
import time
import torch
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
        self.model = TabNetClassifier(device_name='cpu', optimizer_params=dict(lr=8e-3), verbose=0)

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
    ('xgb', XGBClassifier(tree_method='hist', predictor='cpu_predictor', n_estimators=300, max_depth=10, learning_rate=0.01, subsample=0.8, eval_metric='mlogloss', random_state=42)),
    ('catboost', CatBoostClassifier(task_type='CPU', iterations=300, depth=10, learning_rate=0.01, random_seed=42, verbose=0)),
    ('tabnet', SklearnTabNetClassifier())
]

# Define meta-model (Neural Network MLP)
meta_model = MLPClassifier(hidden_layer_sizes=(1024, 512, 256, 128), activation='relu', solver='adam', max_iter=700, batch_size=128, random_state=42, early_stopping=True)

# Define stacking classifier with passthrough for weighted influence
stacked_model = StackingClassifier(estimators=base_learners, final_estimator=meta_model, cv=5, passthrough=True)

# Define other ML models
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Naive Bayes': GaussianNB(),
    'SVM': SVC(),
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(n_estimators=300, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=300, max_depth=10, learning_rate=0.01, subsample=0.8, eval_metric='mlogloss', random_state=42, tree_method='hist', predictor='cpu_predictor'),
    'Stacked Model': stacked_model
}

if __name__ == "__main__":
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        start_time = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        training_time = time.time() - start_time
        
        results[name] = {
            'Accuracy': accuracy * 100,
            'Precision': precision * 100,
            'Recall': recall * 100,
            'F1 Score': f1 * 100,
            'Training Time (s)': training_time
        }
        print(f"{name} Accuracy: {accuracy * 100:.2f}% | Precision: {precision * 100:.2f}% | Recall: {recall * 100:.2f}% | F1 Score: {f1 * 100:.2f}% | Training Time: {training_time:.2f} sec")
    
    # Print comparison table
    print("\nModel Comparison:")
    for model, metrics in results.items():
        print(f"{model}: Accuracy = {metrics['Accuracy']:.2f}%, Precision = {metrics['Precision']:.2f}%, Recall = {metrics['Recall']:.2f}%, F1 Score = {metrics['F1 Score']:.2f}%, Training Time = {metrics['Training Time (s)']:.2f} sec")
    
    # Plot comparison graphs separately
    metrics_list = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    colors = ['blue', 'green', 'red', 'purple']
    
    for i, metric in enumerate(metrics_list):
        plt.figure(figsize=(10, 5))
        values = [results[m][metric] for m in model_names]
        plt.barh(model_names, values, color=colors[i])
        plt.xlabel(f"{metric} (%)")
        plt.title(f"Model {metric} Comparison (Bar Chart)")
        plt.gca().invert_yaxis()
        plt.show()
    
    # Save models separately
    with open('models/StackedModel.pkl', 'wb') as model_file:
        pickle.dump(stacked_model, model_file)
    with open('models/LabelEncoder.pkl', 'wb') as encoder_file:
        pickle.dump(label_encoder, encoder_file)
