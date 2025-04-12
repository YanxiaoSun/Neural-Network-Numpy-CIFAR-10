import itertools
from tqdm import tqdm

from src.Dataset import load_cifar10, preprocess_data, split_val_set
from src.Loss import CrossEntropyLoss
from src.ThreeLayerNN import ThreeLayerNN
from src.Optimizer import SGDOptimizer
from src.Trainer import Trainer


class GridSearcher:
    def __init__(self, opts, defaults):
        self.combinations = self.generate_combinations(opts, defaults)
        self.results = []

    @staticmethod
    def generate_combinations(hyper_param_opts, hyper_param_defaults):
        """
        根据超参数选项表生成所有超参数组合
        :param hyper_param_defaults: 超参数默认值表
        :param hyper_param_opts: 超参数选项表
        """
        for key in hyper_param_opts.keys():
            if len(hyper_param_opts[key]) == 0:
                hyper_param_opts.pop(key)
        for key in hyper_param_defaults.keys():
            if key not in hyper_param_opts.keys() or len(hyper_param_opts[key]) == 0:
                hyper_param_opts[key] = [
                    hyper_param_defaults[key]]  # 用 hyper_param_defaults 中的默认值填充 hyper_param_opts 中的空选项
        # 生成所有超参数组合
        combinations = []
        for values in itertools.product(*hyper_param_opts.values()):
            combination = dict(zip(hyper_param_opts.keys(), values))
            combinations.append(combination)
        return combinations

    @staticmethod
    def generate_config(combination):
        """
        根据超参数组合生成神经网络结构和优化器参数
        :param combination: 超参数组合
        """
        n_layers = sum([1 for key in combination.keys() if "hidden_size" in key]) + 1
        layer_sizes, activation_functions= [], []
        nn_architecture = []
        if n_layers == 1:
            layer_sizes = [3072, 10]
            activation_functions = ["softmax"]
        elif n_layers > 1:
            layer_sizes.append(3072)
            layer_sizes.append(combination["hidden_size_1"])
            activation_functions.append(combination["activation_1"])
            for i in range(1, n_layers - 1):

                layer_sizes.append(combination[f"hidden_size_{i + 1}"])
                activation_functions.append(combination[f"activation_{i + 1}"])

            layer_sizes.append(10)
            activation_functions.append("softmax")

        optimizer_kwargs = {
            "lr": combination["lr"],
            "reg_lambda": combination["reg_lambda"],
            "decay_rate": combination["decay_rate"],
            "decay_step": combination["decay_step"],
        }
        return layer_sizes, activation_functions, optimizer_kwargs

    def search(self, trainer_kwargs, metric="loss"):
        for combination in tqdm(self.combinations):
            layer_sizes, activation_functions, optimizer_kwargs = self.generate_config(combination)
            
            model = ThreeLayerNN(layer_sizes, activation_functions)  
            optimizer = SGDOptimizer(**optimizer_kwargs)
            loss = CrossEntropyLoss()
            X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_cifar10("/home/add_disk_c/sunyanxiao/Neural-Network-Numpy-CIFAR-10/cifar-10-batches-py")
            # 预处理
            X_train, y_train = preprocess_data(X_train_raw, y_train_raw)
            X_test, y_test = preprocess_data(X_test_raw, y_test_raw)
            X_train, y_train, X_val, y_val = split_val_set(X_train, y_train, val_ratio=0.1)
            trainer = Trainer(model, optimizer, loss, **trainer_kwargs)
            history = trainer.train(X_train, y_train, X_val, y_val, save_ckpt=False, verbose=False) 
            valid_loss, valid_acc = trainer.evaluate(X_val, y_val)
            self.results.append((combination, valid_loss, valid_acc))

        if metric == "loss":
            self.results.sort(key=lambda x: x[1])
        elif metric == "acc":
            self.results.sort(key=lambda x: x[2], reverse=True)
        return self.results
