# Solving Utility Maximization Problems using Quantum Reinforcement Learning


*This project was submitted to QHack Open Hackathon 2022. (Team Name: DynamicQuantumWorld)*

*For full tutorials, visit [here](https://github.com/FinnyLime/Quantum-DSGE/blob/main/Utility%20Maximization%20with%20QRL.ipynb)*

In general markets, the competitive equilibrium, or more generally, Dynamic Stochastic General Equilibrium (DSGE) is characterized by a set of state variables and the consumption and production plans of each agent to maximize the utility. Such utility maximization problem has been traditionally dealt with Lagrangian methods. In this Hackathon project, we demonstrate **a quantum approach to solving utility maximization problem.** Specifically, we employed Quantum Reinforcement Learning to train the policy that determines the agent's actions.

### Model

We first constructed a simplified model of DSGE problem based on the recent work done by researchers at University of College, London [1]. In our model, we assume a single, rational household agent, and a single firm existing in the market. The agent is employed by the firm, and the utility per period is given by $u_t=\textrm{ln}(c_{t})-\frac{\theta}{2}n_t^2$, with the action variables $c_{t}$, representing consumption, and $n_t$, the number of hours worked. We use a discrete timestep over $0\leq t<T$. The price is fixed for $p_{t}=1$ for all times, with no interest rate. The wage is given by $w=1$ for $t<T/2$ and $w=0.5$ afterwards. The agent is also subject to a budget constraint $b_{t+1}=b_t+w_tn_t-p_tc_t$, with the no Ponzi condition $b_T=0$ to preempt unlimited borrowing by the agent. The agent wants to maximize the dicsounted utility $\sum_t \beta^t u_t$ with $\beta=0.97$ and $T=20$. To summarize:

* A single agent and a single firm
* Maximize $\sum_{t=0}^{T} \beta^t u_t$, where $u_t=\textrm{ln}(c_{t})-\frac{\theta}{2}n_t^2$, $\beta=0.97$ and $T=20$
* Budget Constraints $b_{t+1}=b_t+w_tn_t-p_tc_t$; $b_T=0$, where $p_{t}=1$, $w=1$ or $0.5$

There have been some attempts to solve this type of problem using classical machine learning or numerical analysis, but Quantum approach to such problems is absent. We applied Quantum Reinforcement Learning technique to solve the problem. Before we explain our solution, we first present some preliminaries on Reinforcement Learning necessary to understand our approach.


### Reinforcement Learning

In Reinforcement Learning problems, there are states, and actions that the agent perform at each time step. The agent is given some rewards upon performing each action. Our goal is to find a set of states and actions that maximize the total rewards over the entire period. To be more mathematically precise, we have to introduce the concept of [Markov Decision Process](https://en.wikipedia.org/wiki/Markov_decision_process) (MDP). A MDP problem is completely characterized by $(S,A,R,P,\gamma)$, where $S$ is the state space, $A$ is the action space, $R$ represents a reward as a function of the current state and the chosen action, $P$ represents a transition probability given (state, action), and $\gamma$ is the discount factor. The objective function we want to maximize is the value function, the expectation value of total rewards over all times, which is given as follows:
$$V(s)=E\biggl[\sum_{t\geq 0}\gamma^t r_t|s_0=s\biggr].$$

While the value function is a measure of how preferrable each state is, it's actually more important to identify how preferaable each (state, action) pair is. The measure of such preferrability of (state,action) pair is given by the Q-value function, which is defined as
$$Q(s)=E\biggl[\sum_{t\geq 0}\gamma^t r_t|s_0=s, a_0=a\biggr].$$

Machine Learning technique that attempts to systematically learn this Q-value function is known as Q-learning. Q-learning is also a method that we utilized in this hackathon project to solve our problem.

### Quantum Reinforcement Learning

Quantum Reinforcement Learning model is virtually same as its classical counterpart, except for the fact that states are now represented and quantum states. As described in Figure1, there is an agent interacting with the environment, which is characterized by its state. The agent's action influences the environment, and this action is determined by the *policy*, which is chosen to maximize the expected reward over all times. The Quantum Circuit implementing this framework is shown below.

In the above, the reward is calculated through a proper choice of $U_r$ and the measurement operator $M$, and the transition of states over time is given by $\theta_t$, where $\theta_t$ represents the agent's action. $\theta_t$ is trained by a separate Quantum DDPG (Deep Deterministic Policy Gradient) algoritm. After enough amount of training on the policy determining the agent's action $\theta_t$ given $s_t$, we're then equipped with the Quantum Neural Network to easily *tell* someone the best action to maximize their lifetime *reward* or *utility*. Pseudocode of Quantum DDPG algorithm is shown below [1]. Essentially, it's about training the Q-value function $Q_\omega(s,\theta)$ and the policy function $\pi_\eta(s)$.

### Our solution

As we're solving Quantum Reinforcement Learning problem, it's essential to define (1) rewards, (2) actions, and (3) states. Reward is straightforwardly given by the utility function. Actions, as specified in the problem statement, consist of consumption and the number of hours worked. We're now left with properly defining the states, and encoding these states into our Quantum Circuit. We first note that the agent's state is completely characterized by specifying $b_t$ and $c_t$ at all times, since $n_t$ could be calculated from the budget constraints. We now encode our state as $$ \left| \psi \right\rangle = \textrm{cos}\frac{\theta}{2}\left| 0 \right\rangle+e^{i\phi}\textrm{sin}\frac{\theta}{2}\left| 1 \right\rangle,$$ where $b_t=\textrm{tan}\theta$ and $c_t=\textrm{tan}{(\phi/4)}$, with $-\pi/2<\theta<\pi/2$ and $0\leq\phi\leq2\pi$. Through this encoding, we can encode the entirety of our state with a single qubit. Note that the reward is computed as the utility function without resorting to $U_r$ and $M$, and is assumed to be negative infinity for states that do NOT satisfy the budget constraint or $n_t\geq0$. Encoding the state with such Bloch angles also makes the transition of states, $U(\theta_t)$, easy to implement, as this transition can be implemented only with the rotations around $x,y,$ and $z$ axes on the Bloch sphere. With this Quantum Reinforcement Learning architecture and the encoding of states, we employed to Quantum DDPG algorithm to learn the Q-value function and the policy. In the final and last section of this notebook, we present our concrete implementation of Quantum Reinforcement Learning for solving our macroeconomic model.
