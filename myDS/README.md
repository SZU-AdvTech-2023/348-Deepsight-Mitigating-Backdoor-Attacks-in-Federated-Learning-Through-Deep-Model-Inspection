# Federated Learning Client Selection

这是对论文[DeepSight: Mitigating Backdoor Attacks in Federated Learning Through Deep Model Inspection](https://arxiv.org/pdf/2201.00763.pdf)的复现；

这里仅仅在单机上进行使用，使用了`MNIST` 数据集。

## Requirements

> python=3.10
> pytorch=2.0.1
> matplotlib
> sklearn
> hdbscan

## Run

无攻击情况：

> python fa_FL.py  --gpu 0  --epochs 10

有攻击情况：

> python fa_FL_bd.py --attack --num_attacker 4  --gpu 0  --epochs 10

无攻击情况+ deepsight 算法：

> python ds_FL.py --gpu 0  --epochs 5   

有攻击情况下+ deepsight 算法：

> python ds_FL_bd.py  --attack --num_attacker 4 --gpu 0  --epochs 10

更多的超参数请阅读[options.py](utils/options.py)，请自行摸索；

使用GPU的显存不超过 2 GB；

## Structure

```
Repo Root
|---- data        			# 存储数据集
	|---- mnist				# 简单的手写数据集
|---- my_oort				# 具体算法
    |---- model				# 主要是模型相关的代码
        |---- __init__.py  	# 初始化包
        |---- DeepS.py		# DeepSight算法的实现
        |---- Fed.py		# FedAvg
        |---- MnistBackdoor.py		# 后门攻击的实现
        |---- Nets.py		# CNN
        |---- Test.py		# 模型训练集测试
        |---- Update.py		# 本地客户端更新
    |---- save_model		# 存储训练结果模型
    |---- saved_updates		# 存储本地训练结果模型
	|---- util				# 工具
		|---- __init__.py  	# 初始化包
		|---- options.py	# 超参数设置
		|---- sampling.py	# 数据集iid划分
	|---- draw.py			# 绘制损失折线图
	|---- draw2.py			# 绘制acc折线图
	|---- ds_FL.py			# 运行没有攻击情况的DeepSight算法
	|---- ds_FL_bd.py		# 运行存在攻击情况的DeepSight算法
	|---- main_random.py	# 运行随机客户端选择方法
	|---- fa_FL.py			# 无攻击无防御情况
	|---- fa_FL_bd.py		# 有攻击无防御情况
	|---- test_ds_bd.py		# 测试DeepSight算法下的后门准确率
	|---- test_fa_bd.py		# 测试无反应情况下的后门准确率
	|---- README.md			# 。。。
	|---- tttt.py       	# 一些测试    
```

## References

```
Shaoxiong Ji. (2018, March 30). A PyTorch Implementation of Federated Learning. Zenodo. http://doi.org/10.5281/zenodo.4321561
```

```
@article{Rieger2022DeepSightMB,
  title={DeepSight: Mitigating Backdoor Attacks in Federated Learning Through Deep Model Inspection},
  author={Phillip Rieger and Thien Duc Nguyen and Markus Miettinen and Ahmad-Reza Sadeghi},
  journal={ArXiv},
  year={2022},
  volume={abs/2201.00763},
  url={https://api.semanticscholar.org/CorpusID:245650333}
}
```