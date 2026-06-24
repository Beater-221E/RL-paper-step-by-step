# 7 Asynchronous Methods for Deep Reinforcement Learning 2016



## 梗概

作者提出一个异步并行的深度学习框架，也就是后来常说的 **A3C** 。这个框架的特征是 **概念简单，实现轻量化**。提出 **多个actor-learners 并行采样再异步更新共享网络参数** 的训练组织方式。通俗来讲，多个智能体副本同时和环境交互，每个副本都产生自己的经验，然后异步将梯度更新到全局。

作者认为 deep RL 的成功是因为正确的使用了 SGD，本质上 deep RL 的训练还是神经网络优化问题。因此作者认为设计一种更好的 SGD 是 deep RL 的核心。

Deep neural networks 总是不稳定，这是由于

- **non-stationary**

  在监督学习中训练集固定，在整个训练过程数据不会变

  而 RL 中数据是智能体自己产生的，这会导致**数据的分布发生变化**。SGD 默认希望训练样本来自于分布稳定的数据集，在 RL 中梯度方向可能是不断变化的，这样训练就容易震荡

- **correlated**

  智能体在环境中产生的数据通常是连续的，因此前后时间戳的数据可能具有极高的相关性。而在 SGD 理论中预设了一个假设 **样本近似独立**。如果使用相关性极高的数据，网络可能沿着某个局部方向疯狂更新





**DQN** 一定程度上解决了上述两个问题。

- **打散相关性** 

  以前数据根据时间戳等以序列形式出现，使用 **Replay** 使得数据被随机采样

- **减缓非平稳性**

  同样的随机采样，新训练数据集和旧数据集可能存在一定的交叉，这使数据的分布变化没那么剧烈



然而， **Replay** 也要付出代价。

- 随机采样中也有大量的旧数据，这意味着我们使用旧数据训练新 policy，这是一种 **off-policy** 训练策略，因此 **Replya Buffer** 方案只能应用于 **off-policy** 算法。
- 由于要使用大量旧数据，所以要占用大量内存
- 这也导致要想使用 Replay Buffer 还需要额外的流程，增加了开支



作者认为 Replay 实际上使用空间换取稳定性

那么作者提出：有没有一种方案能即解决相关性问题，又能保留 on-policy 方法？



作者提出：既然可以让一个 agent 重复学习多遍历史数据，也可以让多个 agents 同时于环境交互，由于这些 agents 与环境的交互是独立的，天然的去相关性，有很强的并行性，且数据的来源和学习的目标均为 $\pi_t$ （即时的策略） ，因此支持 On-Policy。这就是 **asynchronous advantage actorcritic (A3C)**。





## 相关工作

- **Gorila** 

  Gorila 虽然也是异步，但是其还是没有摆脱使用 Replay Buffer，只是增大了 DQN 的参数量

- **MapReduce RL**

  并行应用于加速大型矩阵运算，而不是多个 agents 并行于环境交互

- **Parallel SARSA**

  其推出了多个 actors各自训练然后周期性同步参数，而不是持续异步更新

- **异步的 Q-learning**

  不同的 worker 异步的更新参数那就涉及到版本问题，使用陈旧的梯度更新或者计算会不会导致发散的问题？作者引用 Tsitsiklis 的结论 **对于异步 Q-learning 即是信息有延迟，只要满足条件，仍然能收敛** 

- **分布式动态规划**

- **遗传算法**





## Background

让我们回顾一下传统的 Actor-Critic 和 RL 问题等的定义

**标准的 RL 问题**

- **状态** $s_t$
- **动作** $a_t$
- **奖励** $r_t$
- **策略** $\pi(a|s)$

**累计回报**
$$
R_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}
$$
这里的 $\gamma$ 是折扣因子，是用来平衡未来预期的奖励对当下考虑的影响



**状态-动作价值函数 Q 函数**
$$
Q^\pi(s,a)
=
E[R_t \mid s_t=s,\; a_t=a]
$$
当前在状态 s 先执行动作 a 然后按照策略 $\pi$ 继续行动，最终能获得多少回报





**状态价值函数 V 函数**
$$
V^\pi(s)
=
E[R_t \mid s_t=s]
$$
当前状态 s，直接按照策略 $\pi$ 行动，能获得多少回报



两者区别在于，Q 函数指定第一步动作，V 函数第一步动作也交给决策决定



**Q-learning**

最优的 Q 有
$$
Q^*(s,a) = \max_{\pi}Q^{\pi}(s,a)
$$
将 Q 值改写为包含神经网络参数 $\theta$ 的形式有近似
$$
Q(s,a;\theta)
\approx
Q^*(s,a)
\tag{1}
$$
损失函数记为
$$
L_i(\theta_i)
=
\left(
r
+
\gamma
\max_{a'}
Q(s',a';\theta_{i-1})
-
Q(s,a;\theta_i)
\right)^2
$$
$s'$ 是 $s$ 的下一个状态



作者认为当前的 Q-learning 是 one-step methods 即只看一步奖励，这样奖励传播太慢，于是作者引入



**n-step Return**
$$
r_t
+
\gamma r_{t+1}
+
\gamma^2 r_{t+2}
+\cdots+
\gamma^{n-1} r_{t+n-1}
+
\gamma^n
\max_a
Q(s_{t+n},a)
$$


**Policy Gradient**

前面的 Q-learning 都是 value-based，而 policy-based 目标是直接学习 $\pi(a|s;\theta)$，最大化期望回报
$$
E[R_t]
$$
Williams(1992) 提出的经典 REINFORCE 更新
$$
\nabla_\theta \log \pi(a_t \mid s_t;\theta)\, R_t
$$
它是下式的无偏估计
$$
\nabla_\theta E[R_t]
$$
引入baseline $b_t(s_t)$ 后
$$
\nabla_\theta \log \pi(a_t \mid s_t)
\left(
R_t - b_t(s_t)
\right)
$$
通常使用价值函数近似
$$
b_t(s_t)
\approx
V^\pi(s_t)
$$




**优势函数**
$$
A(a_t,s_t)
=
Q(a_t,s_t)
-
V(s_t)
$$
由于
$$
R_t
\approx
Q^\pi(a_t,s_t)
$$

$$
b_t
\approx
V^\pi(s_t)
$$

因此
$$
R_t - b_t
$$
可以看作 advantage 的估计



**Actor-Critic**

Actor
$$
\pi(a \mid s)
$$
Critic
$$
V^\pi(s)
$$
利用 Critic 提供的价值估计构造
$$
A(s,a)
=
Q(s,a)-V(s)
$$
来降低策略梯度的方差





## Framework

作者列举了四个算法

- one-step Sarsa
- one-step Q-learning
- n-step Q-learning
- advantage actor-critic

在各个板块，作者想要证明 **异步框架是通用的**，对于上述算法都能实现。所以作者不止想提出单独的 A3C，而是想提出一个针对 RL 的异步框架，这个框架有两个核心思想



- **异步动作学习**

  传统的 DQN 只有一个 Agent 与环境交互，产生的数据是序列化的，高相关性的

  A3C 则是多个 Agents 同时探索，由于梯度来源完全不同，自然的满足了无关性要求

  但是有一个麻烦的小问题，梯度如何更新，由于多个代理要同时更新，最经典的解决办法就是 **加锁**

  但是这样会损失很多性能。因此作者引入了 **Hogwild** 思想

  即啥都不管，每个 agent 可以随意的读取和改写参数

  

  > [!NOTE]
  >
  > **为什么 Hogwild 有效 ？**
  >
  > 在极其庞大的**稀疏**数据集中，大部分更新操作只会修改模型中一小部分的参数。因此，不同节点同时修改同一参数的概率极低，这种无锁覆盖带来的微笑误差在整体上可以被抵消，从而保证了模型最终仍然能收敛到最优解。

  

- **探索不同环境**

  例如，有的 worker 以更激进的策略探索，有的以保守策略



针对不同的算法进行了一些适配性改良

- **Async One-Step Q-learning**

  ![111 (1)_page-0001](C:\Users\17994\Downloads\111 (1)_page-0001.jpg)
  
  每一个线程和**自己的环境副本**交互，每一步计算 Q-learning loss 的梯度，公式没有变，仍是
  $$
  y = r + \gamma \max_{a'} Q(s',a'; \theta^-)
  $$
  但是多个线程共享 **公式（1）** 中的参数 $\theta$ ，更新方式就是上文中的 Hogwild 并且会先**累计梯度** ，然后更新一次，类似于 minibatch，这个做的作用是减少不同线程之间互相改写覆盖的几率，而且更新稳定。
  
  
  
- **Asynchronous one-step Sarsa**

  Sarsa 和 Q-learning 的整体异步框架完全一样，区别在于 target
  
  Q-learning 假设下一步总是选择最优动作，而 Sarsa 使用的是 $a'$ 是在状态 $s'$ 下实际会采取的动作。
  
  这一特征使 Sarsa 是 **on-policy** 算法
  
  
  
- **Async n-step Q-learning**

  这个算法的不同之处在于，它使用 **forward view**

  普通的 TD 是一步步更新，看到 $r_t$ 和 $s_{t+1}$ 就立刻更新 $Q(s_t,a_t)$ 

  而 **forward view** 是先往前走 n 步，拿到一串奖励，再回头更新前面的状态动作对
  $$
  R_t = r_t + \gamma r_{t+1} + ... + \gamma^{n} V(s_{t+n},a')
  $$
  大致的流程是

  1. 当前线程先连续跑最多 $t_{max}$ 步

  2. 收集奖励 $r_t, r_{t+1},...,r_{t+k}$

  3. 如果中途终止就停止

  4. 然后从最后一步往前计算 n-step return

  5. 对这段里的每个 state-action pair 都做更新

     

- **Asynchronous advantage actor-critic （A3C）**

  A3C 同时维护两个函数

  **Actor**  $\pi(a_t|s_t; \theta)$	负责选动作

  **Critic**  $V(s_t; \theta_v)$	  负责评估状态价值

  采样的方式和 n-step Q-learning 一样，而 return 定义为
  $$
  R_t
  =
  \sum_{i=0}^{k-1}
  \gamma^i r_{t+i}
  +
  \gamma^k V(s_{t+k};\theta_v)
  $$
  如果遇到终止状态，就没有最后的 bootstrap 项
  $$
  R_t
  =
  \sum_{i=0}^{k-1}
  \gamma^i r_{t+i}
  $$
  即前 k 步用真实奖励，k 步以后用 V 估计

  更新 actor 时使用
  $$
  \nabla_{\theta}
  \log \pi(a_t \mid s_t;\theta')
  \;
  A(s_t,a_t;\theta,\theta_v)
  $$

  其中 advantage 是（使用 $R_t$ 作为 $Q(a,s)$ 的近似）：
  $$
  A(s_t,a_t)
  =
  R_t
  -
  V(s_t;\theta_v)
  $$
  可以将更新方程改写为
  $$
  \nabla_{\theta}
  \log \pi(a_t \mid s_t;\theta)
  \left(
  R_t - V(s_t;\theta_v)
  \right)
  $$
  更新 Critic 时要让自己的估计接近 return
  $$
  (R_t - V(s_t;\theta_v))^2
  $$
  即最小化 value loss

  有意思的是 Policy 和 Value 不完全是两个独立网络，而是共享底层表示。作者认为，判断该做什么动作和判断当前局面好不好都需要理解同一个状态表示。

  最后作者使用了 **交叉熵正则化**，目标鼓励策略保持一定随机性，防止过早停止
  $$
  H(\pi(s))
  =
  -
  \sum_a
  \pi(a|s)
  \log \pi(a|s)
  $$
  最终 policy gradient 变成 
  $$
  \nabla_{\theta}
  \log \pi(a_t|s_t;\theta)
  \Bigl(
  R_t - V(s_t;\theta_v)
  \Bigr)
  +
  \beta
  \nabla_{\theta}
  H(\pi(s_t;\theta))
  $$
  $\beta$ 控制 entropy 的强度



分析完上述四个算法后，作者开始讨论优化器的选择：为什么用 RMSProp？

原文比较了三种优化器

- SGD with momentum
- RMSProp without shared statistics
- RMSProp with shared statistics

最后发现 shared RMSProp statistics 更稳定

RMSProp 维护梯度平方的滑动平均
$$
g
=
\alpha g
+
(1-\alpha)(\Delta\theta)^2
$$

然后更新
$$
\theta
\leftarrow
\theta
-
\eta
\frac{\Delta\theta}
{\sqrt{g+\epsilon}}
$$
shared 的含义就是所有的 worker 共用一个 g，这样保持全局更新尺度稳定





## Experiments

- 四种异步算法全部稳定训练成功，异步训练框架有效且通用
- 大多数游戏收敛速度快于 DQN
- n-step 方法明显优于 one-step 方法
- A3C 整体表现最好
- 线程数双倍增加，收敛时间近似线性下降



## 结论和讨论

- 当时主流认为 **深度神经网络 + 在线强化学习** 天然不稳定

  DQN 引入了 Replay Buffer 和 Target Network 这些稳定化技巧解决这一问题

  A3C 则指出这些方案并非必须，而关键是要 **去除数据的相关性**

- **异步训练有效**

  参数更新来自多个线程与环境交互的多个状态分布，

  而不是单个轨迹上的强相关样本

  这天然的降低数据的相关性，从而提高了稳定性

- **A3C 比 异步 Q-learning 更有效**

  作者认为 Actor-Critic + n-step return 这个组合才是秘诀

  Q-learning 的更新
  $$
  Q(s,a)
  \leftarrow
  r+\gamma \max_{a'}Q(s',a')
  $$
  存在求 max 操作，会带来估计偏差以及 value bootstrap error

  A3C 直接优化策略，避免了上面两点因此更稳定

  > [!NOTE]
  >
  > **Bootstrap Error**
  >
  > 是指强化学习中用一个不准确的价值估计去更新另一个价值估计时产生并传播的误差
  >
  > 在 TD 或 Q-learning 中，我们不是使用真实回报，而使用
  > $$
  > y = r + \gamma V(s')
  > $$
  > 或者
  > $$
  > y = r + \gamma \max_{a'} Q(s',a')
  > $$
  > 作为训练目标
  >
  > 问题在于 $V(s')$ 和 $Q(s',a')$ 本身也是神经网络预测出来的并不是真值
  >
  > 这意味着后继状态价值估错了，那么更新目标也会带着误差

  

- **异步框架并不依赖于某一个 RL 算法**

  可以在更多模型更大网络上验证





