import pandas as pd
import numpy as np
import math
from sklearn.metrics import confusion_matrix
from utils import computeMean, computeCovariance, computeAccuracy


# Load the dataset
def load_clean_data(file_path):
    df = pd.read_csv(file_path)

    df = df.drop(columns=["Sex"])
    return df


# Transforming the target variable 'Rings' into categorical age groups (Young, Adult, Old)
def transform_target(df):

    # Converting Rings to Age classes
    def get_age_group(rings):
        if rings <= 8:
            return "Young"
        elif 9 <= rings <= 11:
            return "Adult"
        else:
            return "Old"

    # The new categorical column is added
    df["AgeGroup"] = df["Rings"].apply(get_age_group)

    # The original 'Rings' column is dropped
    df = df.drop(columns=["Rings"])

    return df


# Splitting the dataset into training and testing sets
def split_data(df, train_size=0.8, random_state=42):

    # I shuffled the dataset to ensure that the training and testing sets are representative
    #  of the overall data distribution. This is important to prevent any bias in the model training and evaluation.
    df_suffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Calculate the split point
    train_size = int(len(df_suffled) * train_size)

    # Slice the dataframe
    train_df = df_suffled.iloc[:train_size]
    test_df = df_suffled.iloc[train_size:]

    return train_df, test_df


# Calculate the Class Priors
def compute_priors(train_df):
    priors = {}

    # Total no of samples in the training set
    total_samples = len(train_df)

    # Identify the unique classes (Young, Adult, Old)
    classes = np.unique(train_df)

    # Calculate the prior probability for each class
    for cls in classes:
        # count how many samples belong to the current class
        class_count = np.sum(train_df == cls)

        # Calculate P(C) = (Count of Class) / (Total Samples)
        priors[cls] = class_count / total_samples

    return priors, classes


def GaussianPDF(x, mean, covariance):
    diff = x - mean
    inv_cov = np.linalg.inv(covariance)
    spreadProb = -0.5 * np.sum((diff @ inv_cov) * diff, axis=1)
    # for i in range(m):
    #     scores[i] = -0.5 * diff[i].T @ inv_cov @ diff[i]
    determinant = np.linalg.det(covariance)
    return spreadProb - 0.5 * np.log(determinant)


def computeProbabilities(X, Y, X_test):
    priors, classes = compute_priors(Y)
    sigma = computeCovariance(X)

    log_posteriors = {}
    for c in classes:
        mean_c = computeMean(X, Y, c)
        log_prob_x_given_c = GaussianPDF(X_test, mean_c, sigma)
        log_posteriors[c] = log_prob_x_given_c + np.log(priors[c])

    # Returning dictionary of unnormalized log-posteriors
    return log_posteriors


def predict(X_train, Y_train, X_test):
    X_test = np.atleast_2d(X_test)
    log_posteriors = computeProbabilities(X_train, Y_train, X_test)
    predictions = np.array(
        [
            max(log_posteriors, key=lambda c: log_posteriors[c][i])
            for i in range(X_test.shape[0])
        ]
    )
    return predictions, log_posteriors


def evaluate(X_train, Y_train, X_test, Y_test):
    predictions, log_posteriors = predict(X_train, Y_train, X_test)
    accuracy = computeAccuracy(Y_test, predictions)
    cm = confusion_matrix(Y_test, predictions)
    return accuracy, cm


def main():
    # Load the dataset
    df = load_clean_data("data/abalone.csv")

    # Transform the target variable
    df = transform_target(df)

    # Split the dataset
    train_df, test_df = split_data(df)

    # Separate features and target
    X_train = train_df.drop(columns=["AgeGroup"]).values
    Y_train = train_df["AgeGroup"].values
    X_test = test_df.drop(columns=["AgeGroup"]).values
    Y_test = test_df["AgeGroup"].values

    # Evaluate the model
    accuracy, cm = evaluate(X_train, Y_train, X_test, Y_test)

    # Print the results
    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{cm}")

    test = np.array([0.455, 0.365, 0.095, 0.514, 0.2245, 0.101, 0.15])
    predictions, log_posteriors = predict(X_train, Y_train, test)
    print(predictions, log_posteriors)


if __name__ == "__main__":
    main()
