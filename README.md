# BTC-Position-State-Transition-Processing-with-Variable-Length-Random-Time-Window-Based-on-LSTM
基于lstm网络的持仓状态推荐
包含训练，推理，输出三个模块

1.实时数据以及历史数据数据来源于gate,
关于训练数据的标记，
默认以一个历史数据前后15天，共30天的数据做排序，取这三十天位于前15%的价格为开仓点，后15%价格为平仓点，作为标记数据进行训练


2.与github上其余使用了ltsm神经网络的项目不同的是，
为了能更好地拟合开仓状态，这个项目的训练集是移动的，
随时保持训练集和推理集的时间戳差距不大。

3.输入输出实例

4.状态机的使用

5.联系方式
