# Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning

## 梗概

1992 年，Backprop 刚刚兴起，但是正如上一篇也是 1992 年 Watkins 1992 年发表的第二篇 Q-Learning，这个时候的训练只有通过 **动作** 得到 **奖励**，但是 Backprop 需要通过 **监督学习** 进行更新，即： 通过输出与正确的 **标签** 计算 **误差**。作者想到能否直接让 **神经网络通过奖励进行学习**，而这一大类学习策略作者就称之为 **强化学习 Reinforce Learning**。Q-Learning 是 value-based RL 的开端，**而 Reinforce 则正是 policy gradient 的开端**





## 动机

1992 年，强化学习还是一个非常松散的概念，不同研究者做的工作差异很大

- 做优化的人在考虑找到最优点
- 做控制的人只关心如何做决策和规划
- 做神经网络的人关心如何训练网络

而作者认为他们需要一个秦始皇：真正现实世界中的智能体必须同时解决这些问题

未来的人工智能必须是 **神经网络**，**决策**，**探索**，**控制**，**优化** 的集合体



作者在 intro 的第二段提出了三个关键词

- **associative reinforcement learning**

  associative 的含义就是 同一个状态对于不同的输入应该也有不同的输出

  输入和输出应该存在映射关系

- **immediate reinforcement**

  > "immediate reinforcement meaning that the reinforcement provided to the learner is determined by the most recent input-output pair only."

  immediate 的含义是 **奖励** 只依赖最近的输入和输出计算

- **delayed reinforcement**

  与 immediate 相反的概念，只有完成所有动作后才能判断奖励，比如围棋，双方进行到最后手才会结算胜负



那么作者希望神经网络通过奖励直接学习梯度，这该怎么做呢？

> a widely used approach ... is to combine an immediate-reinforcement learner with an adaptive predictor or critic based on temporal difference methods.

当时广泛的处理 Delayed Reward 的主流方法是 **actor + critic**

> The actor-critic algorithms investigated by Barto, Sutton and Anderson ... are clearly of this form, as is the Q-learning algorithm of Watkins.

与我们的 TD Learning （Sutton 1988）和 Q-Learning（Watkins 1989）梦幻联动

作者认为 actor-critic 和 Q-Learning 实际上都是 critic，他们都不直接学习策略



作者最后决定将研究范围缩小到 immediate reinforce，这是因为 Delayed Reward 有一个麻烦的问题：

同样以下围棋为例，执黑棋的棋手获胜，他一共下了200手，问题是这200手棋中，哪些棋对于获胜有贡献，贡献有多大？这就是后来强化学习中著名的 **Credit Assignment Problem 奖励归因问题**

作者认为讨论归因问题使 delayed reinforce 过于复杂，因此转向研究 immediate





## Stochastic Units

传统的神经元总是有固定的输入输出映射，就像函数一样，但是这样会有一个问题

例如，一位打工人有以下策略：周五下班了（状态）-> 去吃火锅（动作）

总是这样的路径就没有办法知道去清吧喝酒，看电影，打游戏（其它动作）好不好了



因此作者提出 **把输出设计成一个随机变量**
$$
Y=
\begin{cases}
A, & p = 0.8 \\
B, & p = 0.2
\end{cases}
$$
那么就既有可能选择动作 A 也有可能选择动作 B，这就是 **探索 Exploration** 机制 

数学上，把输出设计成一个随机变量还有一个好处，作者提出，如果动作是随机产生的，那么：
$$
P(a|x, w)
$$
依赖参数 $w$, where

- a 表示动作
- x 表示当前状态
- w 表示神经网络参数（权重）

于是奖励 $r$ 和期望 $E[r]$ 也依赖 $w$，这样我们就可以计算 $\frac{\partial E[r]}{\partial w} = \nabla E[r]$ 这其实就是后来 **Policy Gradient** 的标准形式





## Characteristic Eligibility

上面那个部分作者得出了，通过构建随机单元得到了 $E[r]$ 对 $w$ 可微的结论

我们有 $w$ -> 决定动作概率 -> 决定奖励

但是 $w$ 是一组参数，我们知道在某个奖励出现后，哪一个参数应该负责

其实弯弯绕绕还是要解决这个上文提到的 **参数归因问题**       damn ！！！



作者定义 
$$
e_{ij}
=
\frac{\partial \ln g_i}
     {\partial w_{ij}}
     
     \tag{1}
$$

- $e_{ij}$：如果当前随机输出发生变化，参数 $w_{ij}$ 对这次输出负责多少

  需要注意的是，这里不是直接定义奖励的责任，而是动作概率的责任

- $g_i$：某个动作被选择的概率

- $w_{ij}$：权重



有的同学要问了，为什么不直接求 $g_i$ 对 $w_{ij}$ 的偏微分，还要加个 ln ？（数学警告 ！！！）

- 直觉上

  对 $g_i$ 取对数之后再微分可以衡量当前这个动作出现的相对可能性增加多少
  $$
  \frac{\partial \ln g_i}
       {\partial w_{ij}} = 
       \frac{\frac{\partial g_i}{g_i}}{\partial w_{ij}}
  $$
  相对而言 $\frac{\partial  g_i}{\partial w_{ij}}$ 智能表示概率的绝对变化，显然相对变化更加合理

- 数学上

  作者希望优化的是 
  $$
  E[r|W]
  $$
  也就是给定网络权重 $W$ 时的期望奖励。论文 Theorem 1 说明：

  对于 Reinforce 形式的算法，平均更新方向与
  $$
  \nabla_W [r|W]
  $$
  同向，当学习率统一时，平均更新正比于这个梯度。

  为了证明过程更加清晰和简单，我们先找一个单个随机动作 $y$ 推导：

  假设
  $$
  g(y;w) = Pr(Y = y|w)
  $$
  奖励为 $r(y)$，那么
  $$
  E[r|w] = \sum_y g(y;w)r(y)
  $$
  对 $w$ 求导
  $$
  \frac{\partial E[r|w]}{\partial w}
  =
  \sum_y r(y)\frac{\partial g(y;w)}{\partial w}
  $$
  有恒等式
  $$
  \frac{\partial g(y;w)}{\partial w}
  =
  g(y;w)
  \frac{\partial \ln g(y;w)}{\partial w} 
  
  \tag{2}
  $$
  带入之后
  $$
  \frac{\partial E[r|w]}{\partial w}
  =
  \sum_y
  r(y)g(y;w)
  \frac{\partial \ln g(y;w)}{\partial w}
  $$
  注意到 
  $$
  \sum_yg(y;w)(·)
  $$
  实际上就是对 $Y \sim g(·；w)$ 的期望，所以
  $$
  \frac{\partial E[r|w]}{\partial w}
  =
  E\left[
  r
  \frac{\partial \ln g(Y;w)}{\partial w}
  \right]
  $$
  这说明 $r \frac{\partial lng(Y;w)}{\partial w}$ 是真实梯度的无偏估计，所以公式 **（1）** 的梯度是无偏的

  ==总结来说，由于数学上 $\nabla g = g \nabla \ln g$  因此 $\nabla E[r] = E[r \nabla \ln g]$，从而只用一次采样到的动作和奖励，就能构造期望奖励梯度的无偏估计==



## Theorem 1

作者解释对于 Reinforce 更新有
$$
\Delta w_{ij} = \alpha_{ij}(r-b_{ij})e_{ij}
\tag{3}
$$
其中

-  $e_{ij} = \frac{\partial \ln g_i}{\partial w_{ij}}$ 
- $\alpha_{ij}$ 是学习率
- $r$ 奖励
- $b_{ij}$ 是基线奖励 

作者希望证明 $E[\Delta w] \propto \nabla E[r]$ 即 Reinforce 不是瞎更新，而是确实优化奖励期望

作者引入了 **expected reinforcement** $\rho(W) = E[r|W]$ 即当前参数下的期望奖励

同样根据概率事件期望的定义
$$
\rho = \sum_Y \Pr(Y)r(Y)
$$
对 w 求导可得
$$
\frac{\partial \rho}{\partial w_{ij}} = \sum_Y r(Y) \frac{\partial \Pr(Y)}{\partial w_{ij}}
$$
这个形式非常眼熟，一眼认出跟上面关于 “为什么要加 ln” 的数学推导相似，直接就是用 **（2）** 进行替换可以得到
$$
\frac{\partial \rho}{\partial w_{ij}} 
=
E[r \frac{\partial \ln \Pr(Y)}{\partial w_{ij}}]
= 
E[re]
\tag{4}
$$
非常 amazing 啊，我们发现 $\partial \rho$ 可以被表示为 $E[re]$  这不正是 **（3）** 中的一部分，那接下来再证明 $E[be] = 0$ 

不就恰能说明公式 **（3）**的更新方向正是期望的梯度方向了嘛

作者接下来进行证明 $E[be] = 0$，恰巧，如果我们计算 $E[e]$
$$
\begin{aligned}
E[e]
&=
\sum_Y P(Y)\frac{\partial \ln P(Y)}{\partial w}
\\
&=
\sum_Y P(Y)\frac{1}{P(Y)}
\frac{\partial P(Y)}{\partial w}
\\
&=
\sum_Y
\frac{\partial P(Y)}{\partial w}
\\
&=
\frac{\partial}{\partial w}
\sum_Y P(Y)
\\
&=
\frac{\partial}{\partial w}(1)
\\
&=
0.
\end{aligned}

\tag{5}
$$
那么对于一个不随 $Y$ 变化的 $b$, $E[be] = E[e] = 0$ ， 根据公式 **（4）（5）** 自然可以得到公式 **(3)** 的更新方向正是与期望的梯度方向一致！



> [!NOTE]
>
> 为什么要使用基线奖励 $b$ ?
>
> 假设平均奖励是 5，某次行动获得的奖励是 6，其实只是比平均水平好一些
>
> 如果直接使用 6 作为奖励，这个信号是很大的，更合理的是使用 $r-b=6-5=1$
>
> 其实这里对数据归一化的思想已经有所体现



