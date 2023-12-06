

进度汇报

**2023.11.2**

任务：找论文，寻找制作延时预测器的方法有哪些，总结相应的优缺点



**2023.11.9** 

主线：参考BRP-NAS论文 实现用GNN预测延时



数据集制作：

本周完成：

1.用cifar10数据集训练一个超网络mysupernet.pt

​       训练结果：    正确率：92.76 推理时间：20.395668268203735

2.对超网络进行剪枝

​       写一个随机剪枝代码，将训练好的超网络进行剪枝，并将剪枝后的模型和相应的模型结构保存

3.将剪枝完成的pt模型进行转换，转换为onnx模型和mnn模型 用于后续实际在边缘设备检测实际运行时间用作数据集

4.在自己电脑上运行450个mnn模型并记录延时

5.实现了将某一个保存的模型结构转化为GNN可以读取的输入张量



下周任务：写一个程序将所保存的模型结构和延时时间制作成一个完整的数据集



**2023.11.16**

本周完成：

1.将所保存的模型结构和延时时间制作成一个完整的数据集

2.写一个dataset将数据集数据读取并用于训练将训练代码跑通                     (实际值-差异)/实际值

初步的训练结果：

50轮次lr=0.001：平均差距0.315224 最大差距1.637433 最小差距0.018864 平均误差比0.819179

50轮次lr=0.01 ： 平均差距0.488729 最大差距1.811288 最小差距0.154991 平均误差比0.719651 

100轮次lr=0.01    平均差距0.395943 最大差距1.718501 最小差距0.062204 平均误差比0.772876



250轮次lr=0.001：平均差距0.399984 最大差距1.722543 最小差距0.066246 平均误差比0.770558

250轮次lr=0.0001：平均差距0.275302 最大差距1.59 5929 最小差距0.002616 平均误差比0.842079

250轮次lr=0.00001：平均差距0.295717 最大差距1.617558 最小差距0.000451 平均误差比0.830369

250轮次lr=0.000001 平均差距0.415268 最大差距1.737826 最小差距0.081529 平均误差比0.761791



500轮次lr=0.0001：平均差距0.273622 最大差距1.594115 最小差距0.004430 平均误差比0.843043

​       由于数据集中训练集与测试集的划分是选取前四分之3为训练集后四分之1为测试集，平均误差比会比较低，因为数据集是按剪枝层数顺序排列的，排在前面的是剪枝层数少的。



下周任务：

1.将论文以方法的方式归类，找引用这篇论文的论文

2.将模型结构列明白 画一张图

**2023.11.23**

  (1)基于查找表的方法

| FBNet: Hardware-Aware Efficient ConvNet Design<br/>via Differentiable Neural Architecture Search | CVPR(2019) | 使用一个延迟查找表模型来估计网络的整体延迟 | 通过对搜索空间中使用的几百个运算符的延迟进行基准测试，我们可以很容易地估计整个搜索空间中1架构的实际运行时间(使用逐层预测器，其通过对模型中的每个操作单独测量的延迟求和来导出延迟) |
| ------------------------------------------------------------ | ---------- | ------------------------------------------ | ------------------------------------------------------------ |
| MnasNet: Platform-Aware Neural Architecture Search for Mobile | CVPR(2019) | 通过在边缘设备上直接执行模型测量延迟       | 我们引入了一种多目标神经架构搜索方法，该方法优化了移动的设备上的准确性和真实世界的延迟。 |
| ChamNet: Towards Efﬁcient Network Design through Platform-Aware Model<br/>Adaptation | CVPR(2019) | 为目标设备构建一个LUT，实现快速延迟估计    | LUT由延迟数据库支持，在数据库中有不同输入维度的真实操作延迟（构建一个延迟查找表） |
| ONCE-FOR-ALL: TRAIN ONE NETWORK AND SPECIALIZE IT FOR EFFICENT DEPLOYMENT | ICLR(2019) | 构建一个查找表预测延迟                     |                                                              |
|                                                              |            |                                            |                                                              |

(2)基于GNN的方法

| BRP-NAS: Prediction-based NAS using GCNs                     | NeurIPS（2020） | 一种基于图卷积网络（GCN）的端到端延迟预测器                  | 使用4层GCN，每层有600个隐藏单元，后面是一个完全连接的层，它生成延迟的标量预测。GCN的输入神经网络模型由邻接矩阵A（不对称，因为计算流被表示为有向图）和特征矩阵X（独热编码）编码。我们还引入了一个全局节点（连接到所有其他节点的节点），通过聚合所有节点级信息来捕获神经架构的图嵌入。GCN可以处理任何一组神经网络模型[郑宁新/BRP-NAS (github.com)](https://github.com/zheng-ningxin/brp-nas#Device-measurement) |
| ------------------------------------------------------------ | --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| A Generic Graph-Based Neural Architecture Encoding Scheme for Predictor-Based NAS | ECCV(2020)      | 提出了GATES，一个基于图的神经架构编码器GATES显着提高了架构性能预测 | GATES模型的操作是传播信息的转换                              |
|                                                              |                 |                                                              |                                                              |
| Bridge the Gap Between Architecture Spaces via A<br/>Cross-Domain Predictor | NeurIPS（2022） | 出了一个跨域预测器（CDP），使用了GCN预测性能                 | 提出了一个渐进的子空间自适应策略，以解决源架构空间和目标空间之间的域差异。考虑到两种建筑空间差异较大，设计了一个辅助空间，使转换过程更加顺畅。 |
| COBRA: ENHANCING DNN LATENCY PREDICTION<br/>WITH LANGUAGE MODELS TRAINED ON SOURCE<br/>CODE | ICLR(2022)      | 基于源代码的图神经网络延迟预测                               | 基于源代码的延迟预测利用一个Transformer编码器来学习短代码段的表示,表示由图卷积网络（GCN）聚合，该图卷积网络捕获算法依赖性并估计所实现的DNN的延迟. |
|                                                              |                 |                                                              |                                                              |
|                                                              |                 |                                                              |                                                              |

(3).基于kernels的方法

| nn-Meter: Towards Accurate Latency Prediction of<br/>Deep-Learning Model Inference on Diverse Edge Devices | MobiSys（2021） | 准确预测DNN模型在不同边缘设备上的推理延迟                    | nn-Meter的关键思想是将整个模型推理划分为内核，即，设备上的执行单元，并进行内核级预测。nn-Meter基于两个关键技术构建：（i）内核检测，通过一组设计良好的测试用例自动检测模型推理的执行单元;以及（ii）自适应采样，以从大空间中有效地采样最有益的配置，从而构建准确的内核级等待时间预测器。 |
| ------------------------------------------------------------ | --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| nnPerf: A Real-time On-device Tool Profiling DNN Inference on Mobile Platforms | SenSys （2023） | 实时的设备上分析器，旨在收集和分析移动平台上的DNN模型运行时推理延迟 |                                                              |
|                                                              |                 |                                                              |                                                              |

（4）基于元学习的方法

| HELP: Hardware-Adaptive Efﬁcient Latency<br/>Prediction for NAS via Meta-Learning | NeurIPS（2021） | 提出了硬件自适应有效延迟预测器（HELP） | 将延迟预测问题形式化为一个few-shot回归任务，即给定一个架构-设备对，估计其延迟，利用每个设备上参考架构的延迟作为参考，提出了一个元学习框架，将经验元学习与基于梯度的元学习相结合，以学习跨多个设备泛化的延迟预测模型。 |
| ------------------------------------------------------------ | --------------- | -------------------------------------- | ------------------------------------------------------------ |
|                                                              |                 |                                        |                                                              |
|                                                              |                 |                                        |                                                              |

（5）基于Transformer的方法

| NAR-Former: Neural Architecture Representation Learning towards Holistic<br/>Attributes Prediction | CVPR(2023) | 通过Transformer架构和自注意力机制，实现对神经网络全面属性的预测 | 提出了一个有效的神经架构表示学习框架，该框架由线性缩放网络编码器，基于transformers的表示学习模型，以及一个有效的模型训练方法与数据增强和辅助损失函数组成。 |
| ------------------------------------------------------------ | ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
|                                                              |            |                                                              |                                                              |



（6）基于回归模型的方法

|                                                              |               |                                                              |                                                              |
| ------------------------------------------------------------ | ------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| FOX-NAS: Fast, On-device and Explainable Neural Architecture Search | ICCV(2021)    | 采用多变量回归分析作为预测器，比深度学习方法减少了时间和数据 | 采用多元回归来生成性能预测因子,与基于深度学习的方法相比，多元回归分析可以更好地理解每个变量对结果的影响，并以更少的数据创建有希望的预测器 |
|                                                              |               |                                                              |                                                              |
| MAPLE: Microprocessor A Priori for Latency Estimation        | CVPR（2022）  | 微处理器先验延迟估计（MAPLE）                                | 提出了一个延迟预测技术，采用了一个紧凑的基于神经网络的非线性回归模型的延迟推理，通过测量10个关键性能指标，包括缓存效率，计算速率和指令数等来表示目标硬件。回归模型是硬件感知和DNN架构感知的，因为它接受硬件描述符和DNN架构编码作为输入。 |
| MAPLE-Edge: A Runtime Latency Predictor for Edge Devices     | CVPR(2022)    | MAPLE针对嵌入式设备的改进版，提出了一种新的使用LPM算法优化边缘运行时间的延迟估计器。 | MAPLE-Edge是一种基于硬件回归模型的LPM方法，可以准确估计深度神经网络架构在看不见的嵌入式目标设备上的推理延迟。精度优于MAPLE和HELP |
| CoDL: Efficient CPU-GPU Co-execution for Deep Learning<br/>Inference on Mobile Devices | MobiSys(2022) | 提出轻量但精确的非线性和并发感知延迟预测。（在cpu和gpu上的精度分别为83.21%和82.69%） | 考虑cpu与gpu协同计算的数据共享开销，考虑由平台特征引起的非线性延迟响应，以数据大小和给定的处理器上执行基本单元的时间使用极轻量线性回归模型来学习非线性模型。 |
| Sniper: cloud-edge collaborative inference scheduling with neural network similarity modeling | DAC (2022)    | 开发了一种基于神经网络相似性（NNS）的非侵入性性能表征网络（PCN），以准确预测DNN的推理时间。 |                                                              |

![15d8f9cc33e21625ceb9978a0b72352](一些图片/15d8f9cc33e21625ceb9978a0b72352.png)

数据集扩充重新划分训练集与预测集的最新结果：

50层lr=0.001     平均差距:0.155869  平均精度:0.927594    精度超过百分之90的数据占比:0.776271 

200 lr=0.001      平均差距:0.149200  平均精度:0.932384    精度超过百分之90的数据占比:0.786441

 下周：.找不以NAS为背景的纯做一个延迟预测器的论文

​           .一种是继续深入用GNN的方式

​             一种是从系统的角度 延迟预测只是其中一部分 

​              需要找温老师深入聊一下（在实验室里的其他设备上测试精度看看有多少，先把延时预测做好再做整体的项目）





11.30-12.7 

在一台台式机、树莓派4、edge上部署测试

台式机i5-13600k
最佳结果：平均差距:0.022470  平均精度:0.940947 精度超过百分之90的数据占比:0.818182

笔记本i7-9750H
平均差距0.164508  平均精度0.926178 精度超过百分之90的数据占比:0.743243

平均差距0.164779  平均精度0.925897 精度超过百分之90的数据占比:0.751131

树莓派4b：
平均差距:1.338747  平均精度:0.900229 精度超过百分之90的数据占比:0.607477

平均差距:1.346650  平均精度:0.898821 精度超过百分之90的数据占比:0.616822

edge：
平均差距:0.356154  平均精度:0.891851 精度超过百分之90的数据占比:0.574661

平均差距0.351642  平均精度0.89193   精度超过百分之90的数据占比:0.622449

平均差距:0.356237  平均精度:0.892061 精度超过百分之90的数据占比:0.565611
