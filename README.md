# Navier Stokes方程反问题

## 概述

Navier-Stokes方程是一组描述流体力学中流体速度和压力变化的偏微分方程。Navier-Stokes的反问题是指，在已知某些流体运动特征（如流量、速度等）的条件下，求解出能够产生这些运动特征的流体性质（如黏度、密度等）和流体边界条件（如壁面摩擦力等）的问题。与正问题（即已知流体性质和边界条件，求解流体的运动特征）不同，反问题的解决需要通过数值优化和逆推算法等方法进行求解。

Navier-Stokes的反问题在工程和科学计算中具有广泛的应用，例如在航空、能源、地质、生物等领域中，可以用于优化流体设计、预测流体运动、诊断流体问题等。尽管Navier-Stokes的反问题非常具有挑战性，但是近年来随着计算机技术和数值方法的发展，已经取得了一定的进展。例如，可以通过使用高性能计算和基于机器学习的逆推算法等技术，加速反问题的求解过程，并提高求解精度。

近年来，深度学习的崛起为解决偏微分方程反问题带来了新机遇。物理信息神经网络（Physics-Informed Neural Networks, PINNs）通过将物理方程（如 Navier-Stokes）以残差形式嵌入神经网络的损失函数中，使模型不仅拟合观测数据，还同时满足物理规律。这种融合了数据驱动与模型驱动的混合方法，尤其适用于小样本、高复杂度的科学计算任务。PINNs 能有效提升模型的泛化能力与可解释性，克服纯数据驱动模型对大规模标注数据的依赖。

## 问题描述

本文研究的核心问题为二维不可压缩 Navier-Stokes 方程的物理参数反演与压力场重建，即基于部分观测到的速度场数据，推断动力黏性系数及不可观测的压力分布。与Navier-Stokes方程不同的是，在Navier-Stokes方程反问题中，存在两个未知的参数。这属于偏微分方程的反问题范畴，其数学模型和约束条件明确，但求解结果不唯一，敏感性高，传统方法难以直接求解。
Navier-Stokes方程的反问题形式如下<img src="file:///C:/Users/gongy/AppData/Roaming/marktext/images/2025-06-15-13-51-54-image.png" title="" alt="" width="550">

其中 u(x, y, t) 与 v(x, y, t) 分别表示速度在 x 和 y 方向的分量，p(x, y, t) 为压力场，θ1为非线性对流项系数（通常为 1），θ2为黏性项系数（对应动力黏性系数ν）。

本模型利用PINNs方法学习位置和时间到相应流场物理量的映射，求解两个参数。

## 环境安装

本案例要求 **MindSpore >= 2.0.0** 版本以调用如下接口: *mindspore.jit, mindspore.jit_class, mindspore.data_sink*。

此外还需要安装 **MindFlow >=0.1.0** 版本。 

## 技术路线

算法的整体技术框架如下所述。

### step1：创建数据集

在本模型中，训练数据和测试数据均从[原数据](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fmaziarraissi%2FPINNs%2Fblob%2Fmaster%2Fmain%2FData%2Fcylinder_nektar_wake.mat)中取样得到。训练数据为圆柱绕流场的稀疏速度观测点，未使用压力数据。

数据集的创建及数据加载相关代码在**dataset.py**文件中实现

### step2：构建模型与优化器

模型采用深度为十层的全连接网络，激活函数是`tanh`函数，每层有20个神经元。<img title="" src="file:///C:/Users/gongy/AppData/Roaming/marktext/images/2025-06-15-12-37-58-1749962274472.png" alt="" width="726">

优化器使用L-BFGS。训练目标是联合优化网络参数和系统物理参数 λ₁、λ₂，因此在构建优化器optimizer时，将两个待求解的未知参数与模型的参数一起放入优化器中训练。<img title="" src="file:///C:/Users/gongy/AppData/Roaming/marktext/images/2025-06-15-12-36-55-1749962207688.png" alt="" width="670">

模型构建及优化器构建的其余具体相关代码在**train.py**中实现。

### step3：构建反问题InvNavierStokes

首先，对相关遍历初始化并设置损失函数为MSE形式。

其次，定义 Navier-Stokes 方程的核心部分，包含三个偏微分方程（PDE）。方法pde()对控制方程进行了定义，分别是两个方向的动量方程与连续性方程。![](C:\Users\gongy\AppData\Roaming\marktext\images\2025-06-15-12-42-46-1749962556365.png)

最后，计算模型的总损失，包括两个部分，分别是pde_loss和data_loss。pde_loss是融入模型的物理约束，根据pde_data计算出的pde残差。data_loss是数据拟合，是模型输出与真实标签计算出的损失。<img title="" src="file:///C:/Users/gongy/AppData/Roaming/marktext/images/2025-06-15-12-40-21-1749962408254.png" alt="" width="698">

具体的相关代码在**inv_navier_stokes.py**中实现。

### step4：模型训练

训练epoch设置为10000，对模型进行10000轮训练。训练目标是联合优化网络参数和系统物理参数 λ₁、λ₂。模型训练过程的具体数据记录在**train.log**中。<img title="" src="file:///C:/Users/gongy/AppData/Roaming/marktext/images/2025-06-15-12-41-31-1749962486550.png" alt="" width="696">

### step5：模型推理及可视化

训练后可对流场内所有数据点进行推理，并可视化相关结果如下。可视化相关代码的实现在**utils.py**中实现

+ l2-error。模型预测的U、V和总误差的L2误差随着训练的进行而逐渐减小，最终趋于稳定。![](C:\Users\gongy\AppData\Roaming\marktext\images\2025-06-15-12-43-08-933c9eaae206bc50e33f6630602e33c.png)

+ 在仅观测流体速度 (u, v) 的条件下，对不可观测的压力场 p(t, x, y)进行预测，并同时识别系统参数 λ₁（对流项系数）和 λ₂（黏性系数）的结果如下。在本实验中，λ₁（和 λ₂的标准值分别为1和0.01。可以看到模型的最终结果基本达到设定值。

+ ![](C:\Users\gongy\AppData\Roaming\marktext\images\2025-06-15-12-43-24-cdd38098aa567e4be31280d60ff3d5f.png)

+ 模型训练过程中预测的U、V速度分量、压力场（P_label和P_predict）以及相应的误差场（U_error、V_error、P_error）的二维热图分布。这些热图反映了在不同时间步和空间位置上变量值的变化，颜色梯度表示变量的幅度。这些结果共同展示了模型在参数识别和场重建方面的性能。动态热力图结果见仓库文件。
