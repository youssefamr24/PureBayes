def compute_mean(values):
    values_mean = sum(values) / len(values)
    return values_mean

def compute_variance(values):
    mean = compute_mean(values)
    cumulative_sum = 0
    for x in values:
        cumulative_sum += (x - mean) ** 2
    values_var = cumulative_sum / len(values)
    return values_var

def compute_accuracy(y_true, y_pred):
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
    return correct / len(y_true)
