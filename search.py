import json

from src.GridSearcher import GridSearcher

hyper_param_defaults = {
    "input_dim": 3072,
    "hidden_size_1": 128,
    "hidden_size_2": 32,
    "output_dim": 10,
    "activation_1": "relu",
    "activation_2": "relu",
    "activation_3": "softmax",
    "lr": 0.05,
    "ld": 0.001,
    "decay_rate": 0.95,
    "decay_step": 6000,
}  # 超参数默认值（主要是神经网络结构和优化器参数）


trainer_kwargs = {
    "n_epochs": 10,
    "eval_step": 100,  
}  


def main():
    hyper_param_opts = {
        "hidden_size_1": [128, 256, 512],
        "hidden_size_2": [64, 32, 128],
        "reg_lambda": [0.05, 0.01, 0.1, 0.005],
        "ld": [0.001, 0.005],
    }
    searcher = GridSearcher(hyper_param_opts, hyper_param_defaults)
    results = searcher.search(trainer_kwargs, metric="loss")
    with open("search_results.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
