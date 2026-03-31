import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import pandas as pd

# Gaussian Naive Bayes Visualization
def plot_feature_distributions(df, target_col='AgeGroup'):
    # Identify the feature columns and the target classes
    features = df.drop(columns=[target_col]).columns
    classes = df[target_col].unique()

    # Plot the distribution of each feature for each class
    for feature in features:
        plt.figure(figsize=(8, 5))

        for cls in classes:
            subset = df[df[target_col] == cls]
            sns.kdeplot(subset[feature], label=cls, fill=True)

        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Density')
        plt.legend()
        plt.show()


# Top Words Visualization
def plot_top_words(X, y, top_n=20):
    # Create a DataFrame for easier manipulation
    data = pd.DataFrame({'tokens': X, 'label': y})
    classes = data['label'].unique()
    # Plot the top words for each class
    for cls in classes:
        words = []
        subset = data[data['label'] == cls]
        # Aggregate all tokens for the current class
        for tokens in subset['tokens']:
            words.extend(tokens)
        
        # Count the frequency of each word and get the top N
        counter = Counter(words)
        most_common = counter.most_common(top_n)

        # Separate the words and their counts for plotting
        words, counts = zip(*most_common)

        
        plt.figure(figsize=(10, 5))
        sns.barplot(x=list(counts), y=list(words))
        plt.title(f'Top {top_n} words in class: {cls}')
        plt.xlabel('Frequency')
        plt.ylabel('Words')
        plt.show()