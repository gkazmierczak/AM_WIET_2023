import numpy as np
from utils import split_matrix


def inverse(A: np.ndarray):
    if A.shape == (1, 1):
        x = 1 / A[0, 0]
        return np.array([[x]])

    A11, A12, A21, A22 = split_matrix(A)
    A11_inv = inverse(A11)
    S22 = A22 - A21 @ A11_inv @ A12
    S22_inv = inverse(S22)
    I = np.eye(A11.shape[0])
    B11 = A11_inv @ (I + A12 @ S22_inv @ A21 @ A11_inv)
    B12 = -1 * (A11_inv @ A12 @ S22_inv)
    B21 = -1 * (S22_inv @ A21 @ A11_inv)
    return np.vstack((np.hstack((B11, B12)), np.hstack((B21, S22_inv))))


A = np.array([[1, 1, 1, 1],
              [2, 2, 3, 1],
              [4, 5, 2, 1],
              [1, 1, 1, 2]])
# A = np.array([[1, 2], [5, 6]])

print(inverse(A))
