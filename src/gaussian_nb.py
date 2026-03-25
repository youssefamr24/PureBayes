import pandas as pd
import numpy as np
import math
from utils import compute_mean, compute_variance

# Load the dataset
def load_clean_data(file_path):
    df = pd.read_csv(file_path)

    df = df.drop(columns=['Sex'])
    return df


# Transforming the target variable 'Rings' into categorical age groups (Young, Adult, Old)
def transform_target(df):

    # Converting Rings to Age classes
    def get_age_group(rings):
        if rings <= 8:
            return 'Young'
        elif 9 <= rings <= 11:
            return 'Adult'
        else:
            return 'Old'
        
    # The new categorical column is added
    df['AgeGroup'] = df['Rings'].apply(get_age_group)

    # The original 'Rings' column is dropped 
    df = df.drop(columns=['Rings'])

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
    classes = train_df['AgeGroup'].unique()

    # Calculate the prior probability for each class
    for cls in classes:
        # count how many samples belong to the current class
        class_count = len(train_df[train_df['AgeGroup'] == cls])
        
        # Calculate P(C) = (Count of Class) / (Total Samples)
        priors[cls] = class_count / total_samples

    return priors


# Compute the mean & variance for each feature per class
def compute_class_statistics(train_df):

    class_stats = {}

    # Identify the unique classes (Young, Adult, Old)
    classes = train_df['AgeGroup'].unique()

    # Identify the feature for all columns except 'AgeGroup'
    features = train_df.columns.drop('AgeGroup')

    for cls in classes:
        class_stats[cls] = {}

        class_subset = train_df[train_df['AgeGroup'] == cls]

        for feature in features:
            values = class_subset[feature].values

            # Calculate mean and variance for the current feature and class
            mean = compute_mean(values)
            variance = compute_variance(values)

            class_stats[cls][feature] = {'mean': mean, 'variance': variance}
    
    return class_stats



# Calculate the Guassian Probability Density Function
# Formula: (1 / sqrt(2 * pi * var)) * exp(-(x - mean)^2 / (2 * var))
def calculate_gaussian_likelihood(x, mean, variance):

    # To prevent division by zero
    eps = 1e-9
    if variance < eps:
        variance = eps
    
    # (1 / np.sqrt(2 * pi * np.var))
    coefficient = 1.0 / math.sqrt(2.0 * math.pi * variance)

    # exp(-(x - mean)^2 / (2 * var))
    exponent_value = -((x - mean) ** 2) / (2.0 * variance)

    return coefficient * math.exp(exponent_value)


