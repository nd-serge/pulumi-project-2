import pickle
import argparse
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from utils import load_data, split_data, train_model, evaluate_model, save_model


def training_flow(data_path, model_path):
    print("Starting the training process...")
    
    try:
        
        # Load data
        X, y = load_data(data_path)
        print("Data loaded successfully.")

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = split_data(X, y)
        print("Data split into training and testing sets.")

        # Train the model
        pipe = train_model(X_train, y_train)
        print("Model trained successfully.")

        # Evaluate the model
        evaluate_model(pipe, X_test, y_test)
        print("Model evaluation completed.")

        # Save the model
        save_model(pipe, model_path + "model.pkl")
        print("Model saved successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a logistic regression model.")
    parser.add_argument("--data_path", type=str, default="data.csv", help="Path to the input data CSV file.")
    parser.add_argument("--model_path", type=str, default="./", help="Directory to save the trained model.")
    args = parser.parse_args()
  
    training_flow(args.data_path, args.model_path)

    
        
        # Load data
        