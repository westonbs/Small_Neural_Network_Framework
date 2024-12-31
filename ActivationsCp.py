import cupy as cp

def relu(z):
    return cp.maximum(0, z).astype(dtype=cp.float32)

def sigmoid(z):
    return 1 / (1 + cp.exp(-z))

def softmax(z):
    z_stable = z - cp.max(z, axis=1, keepdims=True)
    exp_z = cp.exp(z_stable)
    return exp_z / cp.sum(exp_z, axis=1, keepdims=True)

def d_relu(z, dJ_da):
    mask = cp.where(z > 0, 1, 0).astype(dJ_da.dtype)  # Ensure consistency in data type
    return mask * dJ_da

def d_test(a, y):
    m = a.shape[0]

    y_true_indices = cp.zeros(a.shape, dtype=cp.float32)
    y_true_indices[cp.arange(m, dtype=cp.int32), y] = 1.0

    return y_true_indices

def d_softmax(a, y):
    m = a.shape[0]

    # Use a sparse encoding of the true labels
    y_true_indices = cp.zeros(a.shape, dtype=cp.float32)
    y_true_indices[cp.arange(m), y.flatten().astype(int)] = 1.0

    return (a - y_true_indices) / m

#todo: learn advanced indexing to finish this
def dJ_softmax(a, y):
    m = a.shape[0]
    y_temp = y.flatten()

    y_true = cp.zeros(a.shape)
    y_true[cp.arange(m), y_temp.astype(cp.int32)] = 1

    return -y_true / (a * m)

def softmax_output(a, y):
    m = a.shape[0]
    max_indices = cp.argmax(a, axis=1)
    y_true_indices = cp.zeros(a.shape).astype(cp.float32)
    y_true_indices[cp.arange(m), max_indices.astype(cp.int32)] = 1.0

    return y_true_indices

def dJ_relu(z):
    return 0