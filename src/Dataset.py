import numpy as np
import pickle
import os

def load_cifar10(data_dir):
    """加载所有训练批次和测试批次"""

    train_data, train_labels = [], []
    for i in range(1, 6):
        batch_file = os.path.join(data_dir, f'data_batch_{i}')
        with open(batch_file, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
        train_data.append(data[b'data'])
        train_labels.extend(data[b'labels'])

    test_file = os.path.join(data_dir, 'test_batch')
    with open(test_file, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    test_data = data[b'data']
    test_labels = data[b'labels']
    
    return np.vstack(train_data), np.array(train_labels), test_data, np.array(test_labels)

def preprocess_data(data, labels):
    """数据预处理"""
    data = (data.astype(np.float32) / 255.0 -0.5) * 2.0
    # One-hot编码标签
    one_hot_labels = np.eye(10)[labels.reshape(-1)]
    return data, one_hot_labels

def split_val_set(X, y, val_ratio=0.2):
    """分割验证集"""
    n = len(X)
    indices = np.random.permutation(n)
    val_size = int(n * val_ratio)
    return X[indices[val_size:]], y[indices[val_size:]], X[indices[:val_size]], y[indices[:val_size]]

if __name__ == "__main__":
    train_data, train_labels, test_data, test_labels = load_cifar10("../cifar-10-batches-py")
    import pdb;pdb.set_trace()
