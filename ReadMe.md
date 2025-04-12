## Neural-Network-Numpy-CIFAR-10

### 环境配置
```
pip install numpy
```

### 数据集下载

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)下载到``cifar-10-batches-py``目录下

### 训练

```
python train.py
```
可以修改模型超参``layer_sizes``和``activation_functions``,其中training_history.png为训练loss，accuracy的记录

### 测试

```
python test.py
```

### 参数可视化
```
python visualize.py
```

### 参数组合搜索

```
python search.py
```

