import os
import re
import math
from collections import defaultdict, Counter

# -----------------------------
# Stop Words (simple list)
# -----------------------------
STOP_WORDS = set([
    "i", "me", "my", "you", "he", "she", 'the','is','at','on','and','a','an','to','of','in','that','this','it','for','with','as','was','were','be','by','are','from','or','but'
])

# -----------------------------
# 1. Text Preprocessing
# -----------------------------

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()

    # remove stop words
    tokens = [w for w in tokens if w not in STOP_WORDS]
    bigrams = [tokens[i] + '_' + tokens[i+1] for i in range(len(tokens)-1)]
    return tokens + bigrams
    

# -----------------------------
# 2. Load Dataset
# -----------------------------

def load_data(path):
    data = []
    labels = []
    for label in ['pos', 'neg']:
        folder = os.path.join(path, label)
        for file in os.listdir(folder):
            with open(os.path.join(folder, file), 'r', encoding='utf-8') as f:
                text = f.read()
                data.append(preprocess(text))
                labels.append(label)
    return data, labels

# -----------------------------
# 3. Train Multinomial NB
# -----------------------------

class MultinomialNB:
    def __init__(self):
        self.vocab = set()
        self.class_word_counts = defaultdict(Counter)
        self.class_counts = Counter()
        self.class_priors = {}
        self.word_probs = {}

    def fit(self, X, y):
        total_docs = len(y)

        for tokens, label in zip(X, y):
            self.class_counts[label] += 1
            self.class_word_counts[label].update(tokens)
            self.vocab.update(tokens)

        V = len(self.vocab)

        # priors
        for c in self.class_counts:
            self.class_priors[c] = math.log(self.class_counts[c] / total_docs)

        # likelihoods (Laplace smoothing)
        for c in self.class_counts:
            total_words = sum(self.class_word_counts[c].values())
            self.word_probs[c] = {}

            for word in self.vocab:
                count = self.class_word_counts[c][word]
                prob = (count + 1) / (total_words + V)
                self.word_probs[c][word] = math.log(prob)

    def predict(self, X):
        predictions = []

        for tokens in X:
            scores = {}
            for c in self.class_priors:
                score = self.class_priors[c]
                for word in tokens:
                    if word in self.vocab:
                        score += self.word_probs[c][word]
                scores[c] = score

            predictions.append(max(scores, key=scores.get))

        return predictions

# -----------------------------
# 4. Accuracy
# -----------------------------

def accuracy(y_true, y_pred):
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return correct / len(y_true)

# -----------------------------
# 5. Run
# -----------------------------

if __name__ == "__main__":
    train_path = r"C:\Users\mosat\Desktop\CogA1\PureBayes\aclImdb\train"
    test_path = r"C:\Users\mosat\Desktop\CogA1\PureBayes\aclImdb\test"

    print("Loading training data...")
    X_train, y_train = load_data(train_path)

    print("Loading test data...")
    X_test, y_test = load_data(test_path)

    model = MultinomialNB()

    print("Training...")
    model.fit(X_train, y_train)

    print("Predicting...")
    y_pred = model.predict(X_test)

    acc = accuracy(y_test, y_pred)
    print(f"Accuracy: {acc * 100:.2f}%")

    # Custom test
    sample = [preprocess("This movie was amazing and wonderful")]
    print("Custom Prediction:", model.predict(sample))
