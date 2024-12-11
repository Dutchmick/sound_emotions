# Objective

# Load packages
import numpy as np
import datetime as dt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Define folder locations
folder_interim = "../data/interim/"
folder_models = "../models/"

# Function to create a training and test dataset
def prepare_data(embeddings, clusters, test_size=0.2, random_state=10):
    return train_test_split(embeddings, clusters, 
                            test_size=test_size, 
                            random_state=random_state, 
                            stratify=clusters)

# Function for hyperparameter tuning
def train_model(X_train, y_train):
    # Define a Random Forest model with hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }
    model = RandomForestClassifier(random_state=10)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

# Function to evaluate the model performance
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Happy", "Sad", "Angry", "Calm"])
    return accuracy, report

# Function that combines data prep + training + model evaluation
def main_workflow(embeddings, clusters):
    # Prepare the data
    X_train, X_test, y_train, y_test = prepare_data(embeddings, clusters)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    accuracy, report = evaluate_model(model, X_test, y_test)
    
    print(f"Model Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    
    return model

# Load data
embeddings = np.load(folder_interim+'embeddings_array.npy')
clusters = np.load(folder_interim+'manual_labels.npy')

# Create and evaluate model
model = main_workflow(embeddings, clusters)

# Export the model to a file
date_time = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
model_name = date_time + '_model.pkl'
with open(folder_models+model_name, 'wb') as f:
    pickle.dump(model, f)