# 自动驾驶小白说:动手学运动规划(Motion Planning)

🧙课程配套教程:[自动驾驶小白说:动手学运动规划](https://www.helloxiaobai.cn/article/bmp)

🏰官网:[自动驾驶小白说](https://www.helloxiaobai.cn/)! 欢迎Star，Follow，Share三连!

🌠代码配合官网教程食用更佳！   

🚀**知乎**, **微信公众号, 知识星球** 全平台同号！

欢迎Star，Follow，Share三连!

# 1. 安装说明
## 1.1 下载代码

```bash
git clone https://github.com/Hello-Xiao-Bai/Planning-XiaoBai.git
```

## 1.2 安装依赖

- 直接在默认环境安装

如果你的Ubuntu环境较简单,不需要有很多的Python环境,那么可以尝试直接安装

```bash
pip3 install -r requirements/requirements.txt
```

- 通过conda环境(推荐)

需要先安装[miniconda](https://docs.anaconda.com/miniconda/)并且最好给conda换源,安装如下:

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
~/miniconda3/bin/conda init bash
# 结束后退出终端,然后重新打开
```

conda准备好之后,创建conda环境

```bash
cd Planning-XiaoBai
conda env create -f requirements/environment.yml
```

之后进入conda环境即可

```bash
 conda activate Planning-XiaoBai
```

## 1.3 运行代码

大部分算法章节，我们都提供了对应的代码实现和可视化。章节后缀.c代表对应的算法实现，以**1.1 车辆运动学:自行车模型**为例，**1.1.c 车辆运动学:自行车模型代码解析**即为它的代码实现。

所有的代码都在**Planning-XiaoBai**文件夹目录执行，以1.1节为例:

```bash
cd Planning-XiaoBai
python3 tests/basic/kinematic_test.py
```

你会看到相应的可视化结果
![BicycleModel1](https://github.com/user-attachments/assets/b2d8ef95-6c78-4f9a-9418-f33647dc7512)


# 2. 内容介绍
自动驾驶运动规划算法是自动驾驶技术中的核心部分，它负责在复杂多变的交通环境中,为自动驾驶汽车规划出安全、高效的行驶轨迹。运动规划模块需要综合上游的信息(感知, 预测, 定位, 地图, 决策等), 考虑车辆的**安全性**、**运动学**、**舒适性**等要求，输出一条合理的轨迹。

本课程会以**理论配合代码**的形式, 讲解基础的运动规划算法以及相关知识. 为了让读者能够在**实战**中真正掌握算法, 我们给绝大部分算法章节, 配置了对应的Python代码以及可视化展示. 代码主要参考了[PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics)等开源项目.

课程需求:

- 基本的数学知识
- 基本的python语言知识
- Ubuntu 18.04或以上的电脑环境

各章节大体内容如下:

## 第0章 引言

主要介绍本课程的主要内容, 如何配置代码运行环境, 以及代码运行说明.

## 第1章 运动规划基础知识

学习运动规划算法之前, 我们对一些常用, 基础的前置知识进行介绍,  以便于读者顺利学习之后的运动规划算法.

- 车辆运动学: 自行车模型和阿克曼转向模型
- 碰撞检测算法: SAT, GJK
- Frenet坐标系

## 第2章 常见的曲线表达形式

主要介绍规划算法中常见的曲线:

- 基于5次多项式的参数方程曲线(Quintic Polynomial)
- 3次样条曲线(Cubic Spline)
- 贝塞尔曲线(Bézier Curve)
- 3次螺旋线(Cubic Spiral Curve)
- Dubins曲线
- Reeds Shepp曲线

## 第3章 基于采样的规划算法

主要介绍基于采样的各类规划算法，以便在不同的场景选择最合适的算法。

- 随机性采样算法：PRM，RRT家族（Basic RRT, Goal based RRT, RRT Connect, 以及RRT Star等）
- 确定性采样算法：基于控制空间的采样，基于状态空间采样

## 第4章 基于搜索的规划算法
主要介绍基于搜索的各类规划算法:

- 图搜索基础:DFS, BFS
- Dijkstra算法, A*算法
- Hybrid A*算法
## 第5章 基于优化的规划算法
主要介绍基于数值优化及相关规划算法:

- 数值优化基本概念
- 梯度下降法,牛顿法
- 线搜索方法
- QP优化
- 基于PiecewiseJerk的路径优化方法
