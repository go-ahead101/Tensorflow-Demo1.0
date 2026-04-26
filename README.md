# 神经网络基础与 TensorFlow 实战 — 代码说明

本目录为课程相关练习代码，涵盖用 **NumPy** 手写前向/反向传播、**TensorFlow/Keras**、**PyTorch** 实现回归，以及常见 **激活函数** 可视化。

## 环境依赖

运行前请安装（版本可按本地环境调整）：

| 用途 | 包 |
|------|----|
| 波士顿房价相关脚本 | `numpy`、`pandas`、`scikit-learn`、`matplotlib` |
| `tensorflow_boston.py` 等 | `tensorflow` |
| `pytorch_boston.py` | `torch` |
| 激活函数示例 | `numpy`、`matplotlib` |

可使用：

```bash
pip install numpy pandas scikit-learn matplotlib tensorflow torch
```

## 数据文件

以下脚本会读取 **`housing.csv`**（波士顿房价，无表头、空格分隔）：

- `numpy_boston.py`
- `tensorflow_boston.py`
- `tensorflow_boston_dataparallel.py`
- `pytorch_boston.py`

请将该数据文件放在**运行时的当前工作目录**（与脚本同级最稳妥）。若数据在上级目录，可复制到本目录，或在包含 `housing.csv` 的目录下执行 `python` 并写出脚本路径。

## 脚本说明

| 文件 | 说明 |
|------|------|
| `numpy_forward.py` | 小型网络前向传播示例（Sigmoid 等，固定权重演示） |
| `numpy_model.py` | 纯 NumPy 实现两层网络、随机数据上的训练与 loss 曲线 |
| `numpy_boston.py` | 纯 NumPy 在波士顿房价上的训练（标准化 + ReLU + MSE） |
| `tensorflow_boston.py` | Keras `Sequential`：13→10(ReLU)→1，训练并绘制 loss 与预测对比 |
| `tensorflow_boston_dataparallel.py` | 同上结构，使用 `tf.distribute.MirroredStrategy` 多 GPU 数据并行（需多卡环境） |
| `pytorch_boston.py` | PyTorch 实现与 TensorFlow 版类似的网络与可视化 |
| `activation_function/sigmod1.py` | Sigmoid 函数与图像 |
| `activation_function/tanh1.py` | Tanh 函数与图像 |
| `activation_function/relu1.py` | ReLU 函数与图像 |

## 运行示例

在包含 `housing.csv` 的当前目录下：

```bash
python tensorflow_boston.py
```

不依赖外部数据文件的示例：

```bash
python numpy_model.py
python numpy_forward.py
python activation_function/relu1.py
```

## 说明

- 各脚本中注释为中文，便于对照教材理解流程。
- `matplotlib` 出图在部分环境需图形界面；无界面服务器可改为 `plt.savefig(...)` 或关闭弹窗相关代码（按需求自行调整）。
