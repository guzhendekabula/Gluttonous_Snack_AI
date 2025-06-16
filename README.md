# 基于强化学习与神经网络的ai贪吃蛇
## 项目背景
这是一款基于强化学习与神经网络的ai贪吃蛇游戏。在这一程序中，我们可以先训练小蛇的动作模型，并将其保存起来。在真正游玩的时候，再将这个已经训练好的模型导出，作为判断下一个动作的方法。

## 开发工具
语言：Python(3.12)

库：pytorch、pygame、collections、random、numpy、os、matplotlib.pyplot、IPython、sys

环境：可以自己搭建一个含有上述库函数的python环境

## 文件解读
1. snake_env.py：运行AI的游戏环境依赖代码文件
2. Snake_agent.py：ai贪吃蛇的代理（训练）
3. helper.py：将每次训练的得分情况以及总的平均分通过图像可视化
4. snake_ai.py：训练、游玩贪吃蛇
5. snake_human：通过键盘玩的贪吃蛇（与该项目无关）
6. model/model.pth：训练的模型

## 运行
1. 导入项目
2. 更改snake_env.py中加载图片的路径参数
3. 对于snake_ai.py，如果想训练，则需要将train()的注释去掉，同时注释play()；如果想游玩则相反，将play()的注释去掉，同时注释train()。

![image](https://github.com/user-attachments/assets/17dc76fe-da7b-42bb-9d7b-640a1f62e950)

## 神经网络与强化学习部分————snake_agent.py
snake_agent.py是代理编码部分。分为三个部分，神经网络、深度Q学习训练器和AI本体，通过引入pytorch模块完成三个类的编码。
### 神经网络
Linear_Qnet是神经网络类，继承于module类。这个神经网络的结构很简单：输入层，一个隐藏层和输出层。输入层参数为20个（env中的20个参数），输出层参数为3个，分别代表直走，左转和右转。__init__函数即为神经网络的搭建，将神经网络初始化并设置好参数，用ReLU函数作为激活函数。而forward函数则为一次前向传播。
### 深度Q学习
训练器所使用的基本逻辑是深度Q学习，是一种结合了深度学习和强化学习的算法，用于解决离散动作空间下的马尔科夫决策过程问题。在训练器的初始化当中设置了优化器和损失函数（我们采用的是均方误差），并定义了一些参数，如学习率、神经网络各层的维度等等。我们还用到了集成学习的方法进行训练。在Snake_agent.py中，我们设置了两个神经网络，一个网络是训练网络，而另一个网络则是目标网络。这两个神经网络模型的结构完全相同，但是权重参数不同；每训练一段之间后，训练网络的权重参数才会复制给目标网络。这样能保证真实值Qtarget(st,a)的估计不会随着训练网络的不断自更新而变化过快。

train_step就是实际进行训练的函数。首先将各个参数转换为tensor格式便于运算处理。首先通过self.model函数计算给定状态的Q值（一个或多个动作对应的Q值向量），然后使用gather函数根据执行的动作（action）索引出对应的Q值，并使用squeeze去掉多余的维度。使用目标网络（self.target_model）计算下一个状态的Q值，通过detach()确保在计算梯度时不会考虑目标网络的参数。然后，使用max函数找到最大Q值，这通常对应于最佳动作。根据强化学习的公式，计算目标Q值，其中self.gamma是折扣因子，用于控制未来奖励的重要性。如果done为真（序列结束），则不考虑未来奖励。计算公式：target = (reward + gamma * Q_value_next * (1 - done))清除之前的梯度，并计算预测Q值与目标Q值之间的损失。通过调用loss.backward()反向传播损失，并使用优化器（self.optimizer）更新模型参数。
### AI本体
agent类中，构造函数同样是将状态数量、动作数量、最大探索次数等的参数传入类中，并初始化训练器和经验回放内存和计数器。这里我们设置最大探索次数为100次，学习率为0.001，折扣因子为0.9，隐藏层为128。

Remember函数将一个元组（包含状态、动作、奖励、下一个状态和是否完成）添加到经验回放内存中。Train_long_memory函数则是从经验回放内存中随机采样一批数据（如果内存中的数据少于batch_size，则使用全部数据），设置256为每一个小批量，通过小批量来训练以达到更好的效果。将采样得到的数据（状态、动作、奖励、下一个状态和是否完成）转换为numpy数组。调用trainer的train_step方法来训练模型。

函数get_action根据当前状态state和游戏数量n_game来选择动作。使用ε-greedy策略：根据epsilon值决定是随机选择动作还是根据模型的预测选择最优动作。如果探索（explore为真）且随机数小于epsilon，则根据模型预测的概率分布随机选择动作。否则，选择模型预测的最优动作（即预测值最大的动作）。

## 结果分析
当训练300次时，平均得分可以稳定在40以上，如图（黄线为平均值）：

<img width="195" alt="image" src="https://github.com/user-attachments/assets/dd7e1dac-338d-45b0-9313-906537f08c8b" />

当训练1000次时，平均分可以稳定在65以上，如图（黄线为平均值）：

<img width="198" alt="image" src="https://github.com/user-attachments/assets/be27b5d3-0e5a-45e2-bd9c-78c12a04632d" />

## 运行时截图
<img width="415" alt="image" src="https://github.com/user-attachments/assets/ba55d666-3e27-4e73-963b-0c930b6d71d9" />

<img width="197" alt="image" src="https://github.com/user-attachments/assets/abbc92a3-b6d2-413e-a2ac-154465e98e50" />   <img width="186" alt="image" src="https://github.com/user-attachments/assets/afcf3991-d51f-45cb-8002-db203f4981ae" />

可以看到，游玩时，小蛇的得分也是越来越高，并且比训练时得分的增长要快很多，最后每轮大概可以稳定在60分以上。



