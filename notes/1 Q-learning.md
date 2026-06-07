# Q-Learning

## 概要

Q-learning 是一种 **model-free 的强化学习** 和 **异步动态规划 （asynchronous dynamic programming）**方法，使智能体能够在 **受控 Markov** 环境中学习如何最优行动。所谓的 **Q** 就是 (state, action)，状态-动作对，Q-Learning 的目标就是 **对每个状态动作对打分，然后不断修正这个分数**。论文证明了 Q-Learning 会以概率 1 收敛到最优 Action Value。



**Contributions:**

- Q-Learning 收敛性证明
- Q-Learning 系统化描述
- 讨论其拓展形式



> [!NOTE]
>
> **受控 Markov 环境**
>
> 普通的 Markov 链状态转移概率 $P(s' | s)$ 固定，例如
>
>  Sunny → Rainy = 0.3     Sunny → Sunny = 0.7
>
> 这就是标准的 Markov Process
>
> 而论文中的 Controlled Markov Process 加入了智能体
>
> 状态 s 时，agent 选择了动作 a，环境发生了变化，状态转移到了 s’
>
> 此时的状态转移变成了 $P(s' | s, a)$
>
> 这就是 **控制马尔可夫过程** 或者现在更常见的称呼 **Markov Decision Process （MDP）**



> [!NOTE]
>
> **异步动态规划**
>
> 想了解异步动态规划，首先我们先要了解什么是 **同步动态规划 （Synchronous DP）**
>
> 举个例子 
> $$
> V_{k+1}(s) = \max_a [R + \gamma \sum P(s'|s,a)V_k(s')]
> $$
> s 是一组状态 {$s_1, s_2, s_3, ... , s_n$}，每次根据更新规则这一组状态总是同时被更新，这就是同步的含义
>
> 问题在于，现实中状态空间巨大，比如围棋约有 $10^{170}$ 个状态，这时我们不可能全部更新再进行下一轮
>
> 异步动态规划应运而生，在同步动态规划的基础上，每个状态不再同时被更新，而是 **访问到谁更新谁**





## The Task for Q-Learning

### 任务描述

考虑一个智能体，生活在一个 **离散且有限的** 世界中，每个时间步都从 **有限动作** 中选择一个动作。

在时间步 n，

1. agent 观察状态 $x_n$ 

2. 选择动作 $a_n$ 

3. 获得随机奖励 $r_n$ 并根据一定的 **Transition Probability** 进入新的状态 $y_n$
   $$
   Prob[y_n=y|x_n,a_n] = P_{x_ny}[a_n] = P(s'|s,a)
   \tag{1}
   $$

Agent 的目标是寻找最优策略，使长期折扣奖励最大
$$
V^{\pi}(x) = R_x(\pi(x)) + \gamma \sum_y P_{xy}[\pi(x)]V^{\pi}(y)
\tag{2}
$$
这就是 **Bellman Equation**，状态价值 = 立即奖励 + 未来状态价值

只不过对未来状态价值我们添加了一个 discount $\gamma$



### 最优价值函数

动态规划理论保证，在有限 MDP （状态有限，动作有限）的情况下一定有最优策略 $\pi$ 

**Bellman Optimality Equation**
$$
V^*(x) = \max_a(R_x(a) + \gamma \sum_y P_{xy}[a]V^*(y))
\tag{3}
$$
**Q 函数**
$$
Q^\pi(x,a) = R_x(a) + \gamma \sum_y P_{xy}[a]V^{\pi}(y) 
\tag{4}
$$
其实看的出来 Bellman Equation 和 Q 函数差距并不大，只是他们描述的对象不一样

前者告诉你当前状态有多好，后者告诉你在这个状态 x 下，动作 a 有多好。



因此作者提出：强化学习问题可以转化为学习 Q 函数的问题，无需直接学习策略和环境，只通过学习 Q 来解决强化学习问题，然后最优策略就可以表示为
$$
\pi^*(s) = arg \max_a Q(s,a)
$$


### Q 的更新方式

$$
Q_n(x,a)=
\begin{cases}
(1-\alpha_n)Q_{n-1}(x,a)+\alpha_n\left[r_n+\gamma V_{n-1}(y_n)\right],
& \text{if } x=x_n \text{ and } a=a_n, \\[6pt]
Q_{n-1}(x,a),
& \text{otherwise.}
\end{cases}
\tag{4}
$$

where
$$
V_{n-1}(y) \equiv \max_b \left\{ Q_{n-1}(y,b) \right\} 
\tag{5}
$$


拆开来看，实际上可以简单的理解成 
$$
Q_{new} = (1-a)Q_{old} + \alpha Target
$$
而 $Target$ 正是 Bellman Equation **(2)** 的右半部分

那到达了新状态 y 之后怎么办 ？ 作者在公式 **(5)** 给出了解释：假设未来永远选最优动作 （Greedy）  



> [!NOTE]
>
> **Off-Policy**
>
> Q-Learning 被广泛的认为是 off-policy，我们简单介绍一下 off-policy 和 on-policy是什么意思
>
> 强化学习中有两个概念 
>
> - **行为策略 Behavior Policy** 怎么行动
> - **目标策略 Target Policy** 希望学成什么样
>
> off-policy 的意思就是说 **行为策略 $\ne$ 目标策略**，反之则是 on-policy
>
> 
>
> 公式 **（5）** 正是导致 Q-Learning 是 off-policy 的原因：
>
> 不管以前 Agent 实际执行了什么 Action，
>
> Q-Learning 在更新时都假设：从下一状态开始，未来每一步都选择当前 Q 最大的 Action。
>
> 





## 收敛性讨论

收敛性条件

- 收敛最重要的条件： **对每个状态动作对，都必须被访问无限次**

- 作者也提出：在随机环境下，任何方法都无法在更弱条件下保证找到最优策略

  简单来说，如果不访问所有 state-action pair，就没有信息，所以**必须访问所有 state-action pair**

- **经验不必连续**，零碎的离散的经验也可以训练



### 结论

$$
Q_n(x,a)=
\begin{cases}
(1-\alpha_n)Q_{n-1}(x,a)+\alpha_n\left[r_n+\gamma V_{n-1}(y_n)\right],
& \text{if } x=x_n \text{ and } a=a_n, \\[6pt]
Q_{n-1}(x,a),
& \text{otherwise.}
\end{cases}
\tag{4}
$$

作者不仅证明了 $Q_n(x,a)\rightarrow Q^*(x,a)$ 而且他还证明了 **所有的 state-action pair 的 Q 值都收敛到最优动作价值**



### 条件

这个结论有三个条件（对照公式**（4）**）

1. **Reward 有界**  $|r_n| \le R$ 否则 Q 也将变得无界

2. 学习率 $0 \lt \alpha_n \lt 1$，并且 
   $$
   \sum_i \alpha_{n_i(x,a)} = \infty
   $$
   意思是说 $\alpha$ 的和是发散的，虽然学习率越来越小，但是不能完全消失（$a \not\approx 0$）

3. 平方和必须收敛
   $$
   \sum_i \alpha^2_{n_i(x,a)} \lt \infty
   $$
   累计噪声方差收敛



### 证明

根据公式 **（4）** 不难发现一个问题:  $Q_n$ 每一步都在变化，而且 $r_n, \  y_n$ 随机。这使得直接证明非常困难

聪明的作者没有直接研究 $Q_n$，而是用了一个小巧思：**有没有一个 MDP，它的最优 Q 值刚好等于 $Q_n$**



读到这里，相信读者会很困惑，这么做有什么意义。

但是作者说年轻人不要太急，把你拉住对你说 ”You have no cards“ ，并递给你了一堆牌

![image-20260606191147021](C:\Users\sherr\AppData\Roaming\Typora\typora-user-images\image-20260606191147021.png)

你翻开牌一看，发现作者是这么设计牌的：

- 把训练历史 $(x_t,a_t,y_t,r_t)$ 表示成卡牌
- 把所有的历史经验堆成了牌堆



#### Action Replay Process (APR)

作者提出了这个新过程。APR 的状态定义为 
$$
(x,n)
$$
其中：

- x 是原 MDP 的状态
- n 是卡片编号（并非时间步，而是第 n 张卡，且只允许使用前 n 张经验卡，当前位于状态 x）



这么构造的目的是将 $Q_n(x,a)$ 中的 n 编码进状态空间，证明的目标又转换为了
$$
Q^*(x,a) = Q_n(x,a) = Q^*_{APR}((x,n),a)
\tag{6}
$$


**状态转移**

1. 假设当前 （$x, n$）执行动作 $a$ ，
2. ARP 不去访问真实环境，而是在历史经验里查询 **最近一次出现 （x，$a$）的经验**
3. reward = （x，$a$）（即第 $a$ 次）记录的奖励
4. 下一状态 = （x，$a$）（即第 $a$ 次）记录的下一状态



这也是作者称之为 **Replay** 的原因：只从历史经验中回放

当然这一部分证明跟强化学习就没什么太多关系了，我们就不在此严格的分析证明了，总的来说

**ARP 把 Q-Learning 的学习历史包装成一个 MDP，然后利用已有的 DP 理论来证明 Q-Learning 收敛。**
