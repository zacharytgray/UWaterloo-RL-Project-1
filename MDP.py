import numpy as np

class MDP:
    '''A simple MDP class.  It includes the following members'''

    def __init__(self,T,R,discount):
        '''Constructor for the MDP class

        Inputs:
        T -- Transition function: |A| x |S| x |S'| array
        R -- Reward function: |A| x |S| array
        discount -- discount factor: scalar in [0,1)

        The constructor verifies that the inputs are valid and sets
        corresponding variables in a MDP object'''

        assert T.ndim == 3, "Invalid transition function: it should have 3 dimensions"
        self.nActions = T.shape[0]
        self.nStates = T.shape[1]
        assert T.shape == (self.nActions,self.nStates,self.nStates), "Invalid transition function: it has dimensionality " + repr(T.shape) + ", but it should be (nActions,nStates,nStates)"
        assert (abs(T.sum(2)-1) < 1e-5).all(), "Invalid transition function: some transition probability does not equal 1"
        self.T = T
        assert R.ndim == 2, "Invalid reward function: it should have 2 dimensions" 
        assert R.shape == (self.nActions,self.nStates), "Invalid reward function: it has dimensionality " + repr(R.shape) + ", but it should be (nActions,nStates)"
        self.R = R
        assert 0 <= discount < 1, "Invalid discount factor: it should be in [0,1)"
        self.discount = discount
        
    def valueIteration(self,initialV,nIterations=np.inf,tolerance=0.01):
        '''Value iteration procedure
        V <-- max_a R^a + e

        Inputs:
        initialV -- Initial value function: array of |S| entries
        nIterations -- limit on the # of iterations: scalar (default: infinity)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        V -- Value function: array of |S| entries
        iterId -- # of iterations performed: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''
        
        V = initialV.copy()
        gamma = self.discount
        iterId = 0
        epsilon = np.inf
        doneIterating = False
        
        while not doneIterating:
            V_new = np.empty_like(V)
            for s in range(self.nStates):
                best = -np.inf
                for a in range(self.nActions):
                    reward = self.R[a, s]
                    value = reward + gamma * (np.dot(self.T[a, s, :], V))
                    if value > best:
                        best = value
                V_new[s] = best # Per state update change from the last iteration
            
            epsilon = np.max(np.abs(V_new - V)) # Used to see if epsilon is close enough to target
            V = V_new
            iterId += 1
            
            if (epsilon < tolerance) or (nIterations != np.inf and iterId >= nIterations):
                doneIterating = True
            
        return [V,iterId,epsilon]

    def extractPolicy(self,V):
        '''Procedure to extract a policy from a value function
        pi <-- argmax_a R^a + gamma T^a V

        Inputs:
        V -- Value function: array of |S| entries

        Output:
        policy -- Policy: array of |S| entries'''

        policy = np.zeros([self.nStates], dtype=int)
        
        for s in range(self.nStates):
            best = -np.inf
            for a in range(self.nActions):
                reward = self.R[a, s]
                value = reward + self.discount * (np.dot(self.T[a, s, :], V))
                if value > best:
                    best = value
                    policy[s] = a

        return policy 

    def evaluatePolicy(self,policy):
        '''Evaluate a policy by solving a system of linear equations
        V^pi = R^pi + gamma T^pi V^pi

        Input:
        policy -- Policy: array of |S| entries

        Ouput:
        V -- Value function: array of |S| entries'''

        sIdx = np.arange(self.nStates)
        Rpi = self.R[policy, sIdx]
        Tpi = self.T[policy, sIdx, :]
    
        # Solve linear system: V = (I − γ T)^{-1} R
        A = (np.eye(self.nStates) - (self.discount * Tpi))
        V = np.linalg.solve(A, Rpi)

        return V
        
    def policyIteration(self,initialPolicy,nIterations=np.inf):
        '''Policy iteration procedure: alternate between policy
        evaluation (solve V^pi = R^pi + gamma T^pi V^pi) and policy
        improvement (pi <-- argmax_a R^a + gamma T^a V^pi).

        Inputs:
        initialPolicy -- Initial policy: array of |S| entries
        nIterations -- limit on # of iterations: scalar (default: inf)

        Outputs: 
        policy -- Policy: array of |S| entries
        V -- Value function: array of |S| entries
        iterId -- # of iterations peformed by modified policy iteration: scalar'''

        policy = initialPolicy
        doneIterating = False
        iterId = 0
        
        while not doneIterating and iterId < nIterations:
            V = self.evaluatePolicy(policy)
            newPolicy = self.extractPolicy(V)
            if np.array_equal(newPolicy, policy):
                doneIterating = True
            policy = newPolicy
            iterId +=1

        return [policy,V,iterId]
            
    def evaluatePolicyPartially(self,policy,initialV,nIterations=np.inf,tolerance=0.01):
        '''Partial policy evaluation:
        Repeat V^pi <-- R^pi + gamma T^pi V^pi

        Inputs:
        policy -- Policy: array of |S| entries
        initialV -- Initial value function: array of |S| entries
        nIterations -- limit on the # of iterations: scalar (default: infinity)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        V -- Value function: array of |S| entries
        iterId -- # of iterations performed: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''
        
        V = initialV.astype(float).copy()
        iterId = 0
        epsilon = np.inf
        
        while (iterId < nIterations) and (epsilon > tolerance):
            V_new = V.copy()
            for s in range(self.nStates):
                a = policy[s]
                V_new[s] = self.R[a, s] + self.discount * np.dot(self.T[a, s, :], V)
            epsilon = np.max(np.abs(V_new - V))
            iterId += 1
            V = V_new

        return [V,iterId,epsilon]

    def modifiedPolicyIteration(self,initialPolicy,initialV,nEvalIterations=5,nIterations=np.inf,tolerance=0.01):
        '''Modified policy iteration procedure: alternate between
        partial policy evaluation (repeat a few times V^pi <-- R^pi + gamma T^pi V^pi)
        and policy improvement (pi <-- argmax_a R^a + gamma T^a V^pi)

        Inputs:
        initialPolicy -- Initial policy: array of |S| entries
        initialV -- Initial value function: array of |S| entries
        nEvalIterations -- limit on # of iterations to be performed in each partial policy evaluation: scalar (default: 5)
        nIterations -- limit on # of iterations to be performed in modified policy iteration: scalar (default: inf)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        policy -- Policy: array of |S| entries
        V -- Value function: array of |S| entries
        iterId -- # of iterations peformed by modified policy iteration: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        V = initialV.astype(float).copy()
        iterId = 0
        epsilon = np.inf
        policy = initialPolicy.copy()

        while (iterId < nIterations) and (epsilon > tolerance):
            V, _, epsilon = self.evaluatePolicyPartially(policy, V, nEvalIterations)

            newPolicy = self.extractPolicy(V)
            iterId += 1
            if np.array_equal(newPolicy, policy) or epsilon <= tolerance:
                break
            policy = newPolicy

        return [policy,V,iterId,epsilon]
        