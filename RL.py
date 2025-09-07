import numpy as np
import MDP

class RL:
    def __init__(self,mdp,sampleReward):
        '''Constructor for the RL class

        Inputs:
        mdp -- Markov decision process (T, R, discount)
        sampleReward -- Function to sample rewards (e.g., bernoulli, Gaussian).
        This function takes one argument: the mean of the distributon and 
        returns a sample from the distribution.
        '''

        self.mdp = mdp
        self.sampleReward = sampleReward

    def sampleRewardAndNextState(self,state,action):
        '''Procedure to sample a reward and the next state
        reward ~ Pr(r)
        nextState ~ Pr(s'|s,a)

        Inputs:
        state -- current state
        action -- action to be executed

        Outputs: 
        reward -- sampled reward
        nextState -- sampled next state
        '''

        reward = self.sampleReward(self.mdp.R[action,state])
        cumProb = np.cumsum(self.mdp.T[action,state,:])
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward,nextState]

    def qLearning(self,s0,initialQ,nEpisodes,nSteps,epsilon=0,temperature=0):
        '''qLearning algorithm.  Epsilon exploration and Boltzmann exploration
        are combined in one procedure by sampling a random action with 
        probabilty epsilon and performing Boltzmann exploration otherwise.  
        When epsilon and temperature are set to 0, there is no exploration.

        Inputs:
        s0 -- initial state
        initialQ -- initial Q function (|A|x|S| array)
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- # of steps per episode
        epsilon -- probability with which an action is chosen at random
        temperature -- parameter that regulates Boltzmann exploration

        Outputs: 
        Q -- final Q function (|A|x|S| array)
        policy -- final policy
        '''

        Q = initialQ.astype(float).copy()
        visits = np.zeros_like(Q, dtype=int)
        gamma = self.mdp.discount
        rng = np.random.default_rng()

        for _ in range(nEpisodes):
            s = s0
            for _ in range(nSteps):
                # select action
                if rng.random() < epsilon:
                    a = rng.integers(self.mdp.nActions)
                else:
                    if temperature > 0:
                        prefs = Q[:, s] / temperature
                        prefs -= np.max(prefs)
                        probs = np.exp(prefs)
                        probs /= probs.sum()
                        a = rng.choice(self.mdp.nActions, p=probs)
                    else:
                        qcol = Q[:, s]
                        maxq = np.max(qcol)
                        best = np.flatnonzero(np.isclose(qcol, maxq))
                        a = rng.choice(best)
                
                # sample reward and next state
                r, s_next = self.sampleRewardAndNextState(s, a)

                # Target
                max_next = np.max(Q[:, s_next])
                target = r + gamma * max_next

                # Update
                visits[a, s] += 1
                alpha = 1.0 / visits[a, s]
                Q[a, s] += alpha * (target - Q[a, s])

                # Continue from new state
                s = s_next
        
        # Greedy policy derivation
        policy = np.zeros(self.mdp.nStates, dtype=int)
        for s in range(self.mdp.nStates):
            qcol = Q[:, s]
            maxq = np.max(qcol)
            best = np.flatnonzero(np.isclose(qcol, maxq))
            policy[s] = best[0] if len(best) == 1 else np.random.choice(best)

        return [Q,policy]    