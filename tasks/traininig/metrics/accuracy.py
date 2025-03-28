import numpy as np

def compute_accuracy(
    ground_truths: np.ndarray, 
    predictions: np.ndarray
):
    num_correct = np.count_nonzero(ground_truths == predictions)
    accuracy = num_correct / ground_truths.shape[0]

    return num_correct, accuracy