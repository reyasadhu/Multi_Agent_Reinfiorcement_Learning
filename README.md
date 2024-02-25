**Multiagent Reinforcement Learning Using Q-learning Method in CUDA**

This is a part of the lab assignment for the course ECE 277 at UCSD for Winter 2024.

For the Qlearning code refer the agent.cu file in both folders in the this path src->Qagent.

**Core Problem:**

There is a 2D grid with several mines and a flag. Our goal is to catch the flag. For the first part of the project, we have used a single agent in a 4×4 grid with 2 mines. For the second part, we have used 128 agents in a 32×32 grid with 96 mines.



Description automatically generated](Aspose.Words.fc1c8d3c-4f48-49e4-9446-43e0ee268301.001.png)       

Description automatically generated](Aspose.Words.fc1c8d3c-4f48-49e4-9446-43e0ee268301.002.png)

We have used an asynchronous Q-learning method to obtain the optimal strategy.


**Problem Formulation:**

In Q-learning, we find the best series of action based on agent’s current state.



The Status is the (x,y) coordinates of the agent.

Reward: 1 for catching flag, -1 for stepping on mine and 0 otherwise.

Action: 0 (right), 1(down), 2(left), 3(up) 

The agent will use a Q-table to take the best possible action based on the expected reward for each state in the environment. Here, Q-table is a table of sets of actions and states, and we use the Q-learning algorithm to update the values in the table.

**Initialize:**

We will initialize the Q-table as zeros. We will build the table with columns based on the number of actions and rows based on the number of states.

All agents start from the (0,0) coordinates at each episode. The episode ends when agent reaches one of mines or flag. The agent becomes inactive after that. For multiagent scenario, when remaining active agents are less than 20% of total agents, the episode restarts.

**Choosing action:**

At the start, the agent will choose to take the random action and on the second run, it will use an updated Q-Table to select the action. Choosing an action and performing the action will repeat multiple times until the training loop stops. The first action and state are selected using the Q-Table. 

We use the Epsilon Greedy strategy to choose an action. It is a simple method to balance exploration and exploitation. The epsilon stands for the probability of choosing to explore and exploits when there are smaller chances of exploring.  

At the start, the epsilon rate is higher, meaning the agent is in exploration mode. While exploring the environment, the epsilon decreases, and agents start to exploit the environment. During exploration, with every iteration, the agent becomes more confident in estimating Q-values.

$$Predicted\ class = \ i^{*}(x) = \begin{matrix}
argmax \\
i \\
\end{matrix}P_{Y|X}\left( i \middle| x \right) = \begin{matrix}
argmax \\
i \\
\end{matrix}P_{X|Y}\left( x \middle| i \right)P_{Y}(i)$$

$$ If\ uniform(0,1) < \epsilon $$ 
$$ a = \uniform(0, num\ of\ actions)$$
$$else  $$
$$a = \begin{matrix}
max \\
\alpha^{'} \\ \end{matrix}  Q(S_{n+1}, a^{'})$$  


In code, we decrease the epsilon by 0.1 at each episode. 

**Perform Update:**

After each action, we use this function to update the Q table.

QSn,An+=αRn+1+γmaxα'QSn+1,a'-QSn,An, Rn+1=0αRn+1-QSn,An, Rn+1≠0

Where, $R_{n+1}$ is the reward, $S_{n+1}$ is the next state and $S_{n}$ is the current state. $\alpha$ is the learning rate and $\gamma$ is a discount factor.

**Pseudo -Code:**

Initialize Qs,a=0,∀s∈S,a∈As

Repeat (for each episode):

Initialize S

Repeat for each step of the episode:

`	`Choose A from current state S using policy derived from Q (Epsilon Greedy)

`	`Take action A

`	`Observe next state S’ and R

`	`Update Q(S,A)

`	`S=S’

Until S is terminal

**Source Code:**
'''

int q\_learningCls::learning(int\* board, unsigned int& episode, unsigned int& steps) {

`    `if (m\_episode == 0 && m\_steps == 0) { // only for first episode

`        `env.reset(m\_sid);

`        `agent.init(); // clear action + init Q table + self initialization

`    `} else {

`        `active\_agent = check\_status(board, env.m\_state, flag\_agent);

`        `if (m\_newepisode) {

`            `env.reset(m\_sid);

`            `agent.init\_episode(); // set all agents in active status

`            `float epsilon = agent.adjust\_epsilon(); // adjust epsilon

`            `m\_steps = 0;

`            `printf("EP=%4d, eps=%4.3f\n", m\_episode, epsilon);

`            `m\_episode++;

`        `} else {

`            `short\* action = agent.action(env.d\_state[m\_sid]);

`            `env.step(m\_sid, action);

`            `agent.update(env.d\_state[m\_sid], env.d\_state[m\_sid ^ 1], env.d\_reward);

`            `m\_sid ^= 1;

`            `episode = m\_episode;

`            `steps = m\_steps;

`        `}

`    `}

`    `m\_steps++;

`    `env.render(board, m\_sid);

`    `return m\_newepisode;

}
'''




