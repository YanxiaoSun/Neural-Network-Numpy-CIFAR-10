import matplotlib.pyplot as plt

from src.Dataset import load_cifar10, preprocess_data, split_val_set
from src.Loss import CrossEntropyLoss
from src.ThreeLayerNN import ThreeLayerNN
from src.Optimizer import SGDOptimizer
from src.Trainer import Trainer

DATA_DIR = "cifar-10-batches-py"  # 数据集路径
layer_sizes = [3072, 256, 64, 10]
activation_functions = ["relu", "relu", "softmax"]


optimizer_kwargs = {
    "lr": 0.05,
    "reg_lambda": 0.005,
    "decay_rate": 0.95,
    "decay_step": 6000,
}  # 优化器参数（包括学习率、L2正则化系数、学习率衰减率、学习率衰减步数）

trainer_kwargs = {
    "n_epochs": 100,
    "eval_step": 1,
}  # 训练器参数（包括训练轮数、评估步数）


def main():
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_cifar10(DATA_DIR)
    # 预处理
    X_train, y_train = preprocess_data(X_train_raw, y_train_raw)
    X_test, y_test = preprocess_data(X_test_raw, y_test_raw)
    X_train, y_train, X_val, y_val = split_val_set(X_train, y_train, val_ratio=0.1)
    model = ThreeLayerNN(layer_sizes, activation_functions)  
    optimizer = SGDOptimizer(**optimizer_kwargs)  
    loss = CrossEntropyLoss()  

    trainer = Trainer(model, optimizer, loss, **trainer_kwargs)  
    history = trainer.train(X_train, y_train, X_val, y_val, save_ckpt=True, verbose=True)  
    trainer.save_log("logs/")  
    trainer.save_best_model("models_2/", metric="loss", n=3, keep_last=True)  
    trainer.clear_cache() 

    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss_history'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc_history'], label='Train')
    plt.plot(history['val_acc_history'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()


if __name__ == "__main__":
    main()
