import numpy as np
from src.Dataset import load_cifar10, preprocess_data, split_val_set
from src.Loss import CrossEntropyLoss
from src.ThreeLayerNN import ThreeLayerNN


ckpt_path = "models/model_epoch_35.pkl"


def main():
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_cifar10("./cifar-10-batches-py")
    # 预处理
    X_train, y_train = preprocess_data(X_train_raw, y_train_raw)
    X_test, y_test = preprocess_data(X_test_raw, y_test_raw)
    layer_sizes = [3072, 256, 64, 10]
    activation_functions = ["relu", "relu", "softmax"]
    model = ThreeLayerNN(layer_sizes, activation_functions)  
    model.load_model_dict(ckpt_path)  
    loss = CrossEntropyLoss()

    total_loss = 0
    total_acc = 0

    for i in range(0, len(X_test), 64):
        x_batch = X_test[i:i + 64]
        y_batch = y_test[i:i + 64]
        y_pred = model.forward(x_batch)
        total_acc += np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_batch, axis=1))
        ce_loss = loss.forward(y_pred, y_batch)
        total_loss += ce_loss * len(x_batch)

    test_loss = total_loss / len(X_test)
    test_acc = total_acc / len(y_test)

    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Checkpoint: {ckpt_path} | ")


if __name__ == "__main__":
    main()
