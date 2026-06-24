# 8 Continuous Control with Deep Reinforcement Learning 2016

## 概论

Deep Q-Learning 很成功，但是只能对于**离散有限动作空间**使用，那么作者想能不能推广到**连续动作**。

作者提出了一个 **actor-critic**，**model-free**，**off-policy**，**deterministic policy gradient** 的算法去适配高维的（well，至少在当时算是高维）连续动作空间。

**同样的学习算法，同样的网络结构，同样的超参数** ，作者提出的这一套算法可以在超过 20 个模拟物理任务上面稳健的解决。这说明算法具有一定的通用性。





## 动机

作者认为 AI 的终极目标是 **从原始感知输入直接生成决策**，这意味着**不由人工设计特征**。

这正是 DQN 的成功之处：Deep RL 可以达到人类级别的表现。

但是，尽管 DQN 能处理高维状态，但是不能处理高维动作。这是受困于 DQN 更新算法本身：
$$
y = r + \gamma \max_{a'} Q(s',a')
$$
离散的动作如上下左右，可以直接被列举然后计算最大值。

但连续动作想要计算最大值就变成一个**连续优化问题**，这就是作者为什么提出 DDPG



Q：那把连续动作离散化不就好了？

A：

- **维度灾难**

  即使是非常粗糙的离散化（比如离散成10^3个动作），当这些动作再组合数量级也是恐怖的。

- **精度损失**



是时候拿出 DDPG 了。DPG（Deterministic Policy Gradient） 并不是新算法，首次在 2014 被 Silver 提出。但是 DPG + Neural Network 训练不稳定，所以作者希望在此基础上对其进行优化，使 DPG 像 DQN 一样稳定。作者使用了以下技巧：

- **Replay Buffer**

  类似 DQN，目的是为了打散相关性

- **Target Network**

  同样借鉴 DQN，为了稳定 TD Target

- **Batch Normalization**

  机器人控制状态量尺度差异巨大，BN 可以缓解训练困难

到这里，心急的朋友没有看到 DDPG 在连续动作空间上的适配性改进，吵着赶快搬上来罢！

知道你很急，但你先别急。像往常的综述一样，我们还是要走一遍 “前情提要”



## 问题定义

**状态定义**

依旧老三样

- 状态   $x_t$
- 动作   $a_t$
- 奖励   $r_t$

需要注意的是，环境通常是部分被观测的，POMPDP（Partially Observable Markov Decision Process）中观测 $x_t$ 不一定包含全部信息。比如机器人看到面前的墙，但是不知道墙后是什么。

这里作者还是假设我们是全知的，后文全部按 MDP 讨论。



**随机策略**

DQN 中使用了随机策略 $\pi:S \to P(A)$ ，$P(A)$ 表示动作概率分布，即 $\pi(a|s)$

这时候还没有引入 deterministic policy，因为作者想建立一般情况



**回报定义**
$$
R_t = \sum^T_{i=t} \gamma^{i-t} r(s_i,a_i)
$$
这和之前 A3C 的定义完全一样：未来奖励折扣和





**优化目标**

学习一个 policy 获取
$$
\max_{\pi} J(\pi)
$$


**Q 函数**
$$
Q^{\pi}(s_t,a_t) = E[R_t|s_t,a_t]
$$
当前状态 $s_t$，先执行 $a_t$，然后按照 $\pi$ 继续行动，最后获得多少回报，这是标准的 action value function



**Bellman Equation**

把未来奖励拆开
$$
R_t = r_t + \gamma R_{t+1}
$$
然后代入 Q 定义就得到了
$$
Q^\pi(s_t,a_t)
=
E
\left[
r(s_t,a_t)
+
\gamma
E_{a_{t+1}\sim\pi}
Q^\pi(s_{t+1},a_{t+1})
\right]
\tag{1}
$$
由于策略是随机的，下一步动作 $a_{t+1}$ 要从 $\pi(a|s)$ 采样，因此写作 $E_{a_{t+1}\sim\pi}$ ，必须积分



**Deterministic Policy**

与随机策略的概率分布选择动作不同，Deterministic Policy 策略是固定的 $a = \mu(s)$ ，没有采样

这样 Bellman Equation **公式 1** 就可以被改写为确定版
$$
Q^\mu(s,a)
=
E
\left[
r
+
\gamma Q(s',\mu(s'))
\right]
\tag{2}
$$


作者发现，**期望只依赖于环境**。

随机策略中，期望来自于 **环境随机性** 和 **策略随机性**，而 DPG 使用确定策略，因此只有环境随机性





**Off-Policy**

观察 **公式 2** 里面没有 $\beta(a|s)$ 和 $\pi(a|s)$，只有环境转移

因此数据可以来自于 $\beta$ 而学习目标仍然是 $\mu$，这时行为策略（生成数据）和目标策略（学习最优策略）并不一样，是典型的 off-policy



**Q-Learning Loss**
$$
L(\theta^Q)
=
E
\left[
\left(
Q(s_t,a_t|\theta^Q)
-
y_t
\right)^2
\right]
$$
这是标准 TD Loss，其中 $\theta^Q$ 表示 critic 参数。

目标是为了让 Q 更接近 $y_t$



**Target**
$$
y_t
=
r(s_t,a_t)
+
\gamma
Q\bigl(s_{t+1},\mu(s_{t+1})\mid\theta^Q\bigr)
$$
DQN 中的 max 操作消失了，被 $\mu$ 取代了，这解决了连续动作空间无法枚举的问题







## 算法实现

**更新**

更新时
$$
\nabla_{\theta^\mu} J
\approx
E_{s_t \sim \rho^\beta}
\left[
\nabla_a Q(s,a|\theta^Q)
\big|_{a=\mu(s)}
\,
\nabla_{\theta^\mu}\mu(s|\theta^\mu)
\right]
\tag{3}
$$
从结构上，右侧为
$$
\nabla_a Q \times \nabla_{\theta^{\mu}} \mu
$$
实际上是链式法则 $Q(s,\mu(s))$ 对 $\theta^{\mu}$ 求导，即
$$
\frac{\partial Q}{\partial \theta^\mu}
=
\frac{\partial Q}{\partial a}
\frac{\partial a}{\partial \theta^\mu}
$$
这就是 DPG 和 Policy Gradient 最大的区别

REINFORCE 
$$
\nabla_{\theta} \log \pi (a|s) R
$$
通过采样估计

DPG 直接利用
$$
\nabla_{a} Q
$$
注意到 $E_{s_t \sim \rho^{\beta}}$ 不是 $\rho^{\mu}$ ，状态来自于行为策略 $\rho$ 而不是学习目标 $\mu$ 

这里就解释了为什么 DPG 是 off-policy





**Replay**

由于是 off-policy，所以可以直接使用 Replay Buffer

这里和 DQN 的使用方法完全一样，缓存
$$
(s_t,a_t,r_t,s_{t+1})
$$
训练时随机采样，从而 **去相关性**





**优化**

那么对于优化的目标 $y$，作者认为直接在神经网络使用 Q-learning 是公认的不稳定

原因是 TD Target 依赖自己
$$
y = r + \gamma Q(s',a')
$$
Q 变了 Target 也变了，追逐一个不断移动的目标容易发散

DQN 的方案是每隔 n 步复制一次目标 $Q'$，相当于 n 步前的自己给现在的自己打分

DDPG 的方案则不使用硬复制，而是 soft update
$$
\theta' \leftarrow \tau \theta + (1-\tau) \theta' \\
\text{where} \ \ \tau \ll 1
$$
就是 Target 慢慢的追随 online network



**BatchNorm**

机器人状态尺度差异巨大，使用归一化加速训练
$$
x' = \frac{x-\mu}{\sigma}
$$
**探索**

作者指出，这是连续控制最难的问题

离散动作使用 $\epsilon$-greedy：智能体有 N 个可选的离散动作，在每一步决策时，

- 以 $(1-\epsilon) + \frac{\epsilon}{N}$ 的概率选择最优（Q 最大）动作
- 以 $\frac{\epsilon}{N}$ 的概率选择其它动作



而 DDPG 选择构造一个探索 policy，即 Actor输出 + 噪声
$$
\mu'(s_t) = \mu(s_t) + \mathcal{N}
$$
作者使用 OU Process，Ornstein-Uhlenbech Process

作者专门解释了为什么不使用高斯白噪声：

这是因为物理系统有惯性，通常是随着时间连续变化，OU 噪声具有时间相关性，因此更适合





DDPG 伪代码

![111_page-0001](C:\Users\17994\Downloads\111_page-0001.jpg)



## 实验结论

- DDPG 能解决连续控制问题
- Pixel 也能学
- 针对 DDPG 设计 Target Network 极其重要
- BatchNorm 能解决泛化和学习效率的问题
- Q-learning 容易高估，经检查，在简单任务中比较准确，复杂任务中存在误差，但策略仍学得很好
- 尽管比 DQN 好，但是仍需要大量交互，这是 model-free RL 的通病