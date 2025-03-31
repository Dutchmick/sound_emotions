# Objective
''''
Module uses extracted audio embeddings and assigned labels to create
a predictive model which can be used to predict emotions of new audio files.
'''

# Load packages
import logging
import datetime as dt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import pickle

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

# Function that combines data prep + training + model evaluation
def modelling_workflow(embeddings, clusters, folder_models):
    try:
        logging.info("Initializing model training")
        # Prepare the data
        X_train, X_test, y_train, y_test = prepare_data(embeddings, clusters)
        
        # Train the model
        model = train_model(X_train, y_train)
        
        # Export the model to a file
        date_time = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_name = date_time + '_model.pkl'
        with open(folder_models+model_name, 'wb') as f:
            pickle.dump(model, f)

        logging.info("Model training completed successfully")
        return model, X_test, y_test
    except Exception as e:
        logging.error("Error during model training: %s", e)
        raise
