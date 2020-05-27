# Project 3 - Collaboration and Competition
## Training multiple DeepRL agents to solve the Unity Tennis task

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.


## Implementation
The basic algorithm lying under the hood is an actor-critic method. Policy-based methods like REINFORCE, which use a Monte-Carlo estimate, have the problem of high variance. TD estimates used in value-based methods have low bias and low variance. Actor-critic methods marry these two ideas where the actor is a neural network which updates the policy and the critic is another neural network which evaluates the policy being learned which is, in turn, used to train the actor.



![img](https://cdn-images-1.medium.com/max/2400/1*e1N-YzQmJt-5KwUkdUvAHg.png)



In vanilla policy gradients, the rewards accumulated over the episode is used to compute the average reward and then, calculate the gradient to perform gradient ascent. Now, instead of the reward given by the environment, the actor uses the value provided by the critic to make the new policy update.



![img](https://cdn-images-1.medium.com/max/2600/1*4TRtwlftFmWGNzZde45kaA.png)

[Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/abs/1509.02971) lies under the class of Actor Critic Methods but is a bit different than the vanilla Actor-Critic algorithm. The actor produces a deterministic policy instead of the usual stochastic policy and the critic evaluates the deterministic policy. The critic is updated using the TD-error and the actor is trained using the deterministic policy gradient algorithm.


There are a few techniques which contributed significantly towards stabilizing the training:

- **Fixed targets**: Originally introduced for DQN, the idea of having a fixed target has been very important for stabilizing training. Since we are using two neural networks for the actor and the critic, we have two targets, one for actor and critic each. 
- **Soft Updates**: In DQN, the target networks are updated by copying all the weights from the local networks after a certain number of epochs. However, in DDPG, the target networks are updated using soft updates where during each update step, 0.01% of the local network weights are mixed with the target networks weights, i.e. 99.99% of the target network weights are retained and 0.01% of the local networks weights are added.
- **Experience Replay**: This is the other important technique used for stabilizing training. If we keep learning from experiences as they come, then we are basically observed a sequence of observations each of which are linked to each other. This destroys the assumption of the samples being independent. In ER, we maintain a Replay Buffer of fixed size (say N). We run a few episodes and store each of the experiences in the buffer. After a fixed number of iterations, we sample a few experiences from this replay buffer and use that to calculate the loss and eventually update the parameters. Sampling randomly this way breaks the sequential nature of experiences and stabilizes learning. It also helps us use an experience more than once.

### Multi Agent Deep Deterministic Policy Gradient(MADDPG)
This solution is based on the MADDPG algorithm, using seperate actors and critics for each agents and a shared memory buffer. For Actor and Critic I used two hidden layers with 128 units. For activation I tested Leaky Relu and Relu. Relu showed a better performance. I also tested dropout with p=0.1 but model didn't converge. So I removed it.


### Hyper Parameters

There were many hyperparameters involved in the experiment. The value of each of them is given below:

| Hyperparameter                      | Value |
| ----------------------------------- | ----- |
| Replay buffer size                  | 1e5   |
| Batch size                          | 256   |
| $\gamma$ (discount factor)          | 0.99  |
| $\tau$                              | 1e-3  |
| Actor Learning rate                 | 1e-3  |
| Critic Learning rate                | 1e-3  |
| Epsilon		              | 1.0   |
| The Ornstein-Uhlenbeck noise sigma  | 0.1   |


### Results

The best performance was achieved by **MADDPG** where the reward of +0.5 was achieved in **2585** episodes. I noticed how changing every single hyperparameter contributes significantly towards getting the right results and how hard it is to identify the ones which work. The plot of the rewards across episodes is shown below:


![](scores.png)

---

## Ideas for improvement
- Using Prioritized Replay ([paper](https://arxiv.org/abs/1511.05952)) has generally shown to have been quite useful. It is expected that it'll lead to an improved performance here too.

- Fine tune the parameters

- Compare performance with shared actors, shared critics, shared actors *and* critics.


