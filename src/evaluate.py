
'''
Module provides functions for evaluating the performance of a 
classification model using common metrics such as 
accuracy, confusion matrix, and classification report. 
It is designed to take a trained model and test data as input, 
calculate these metrics, and log the results.
'''

import logging
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Function to evaluate the model performance
def calculate_metrics(model, X_test, y_test):
    '''
    Predicts the labels for the test data using the model and 
    then calculates the accuracy score, confusion matrix, 
    and classification report'''
    target_names=["Happy", "Sad", "Angry", "Calm"]
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    con_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=target_names)
    
    return accuracy, con_matrix, class_report

def log_metrics(accuracy, con_matrix, class_report):
    """
    Function takes the calculated evaluation metrics 
    (accuracy, confusion matrix, and classification report) 
    as input and adds them to the logs
    """
    logging.info("Accuracy Score: %.2f", accuracy)
    logging.info("Confusion Matrix:\n%s", con_matrix)
    logging.info("Classification Report:\n%s", class_report)

def evaluate_model(model, X_test, y_test):
    '''
    Function which combines the model evaluation functions into
    a single module
    '''
    try:
        logging.info("Starting model evaluation")
        accuracy, con_matrix, class_report = calculate_metrics(
            model, X_test, y_test)

        log_metrics(accuracy, con_matrix, class_report)
        return accuracy, con_matrix, class_report

    except Exception as e:
        logging.error("Error during model evaluation: %s", e)
        raise