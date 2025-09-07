from MDP import *

''' Construct simple MDP as described in Lecture 2a Slides 13-14'''
# Transition function: |A| x |S| x |S'| array
T = np.array([[[0.5,0.5,0,0],[0,1,0,0],[0.5,0.5,0,0],[0,1,0,0]],[[1,0,0,0],[0.5,0,0,0.5],[0.5,0,0.5,0],[0,0,0.5,0.5]]])
# Reward function: |A| x |S| array
R = np.array([[0,0,10,10],[0,0,10,10]])
# Discount factor: scalar in [0,1)
discount = 0.9        
# MDP object
mdp = MDP(T,R,discount)

'''Test each procedure'''
[V,nIterations,epsilon] = mdp.valueIteration(initialV=np.zeros(mdp.nStates), nIterations=1000)
print("Optimal Value: ")
print(V, nIterations, epsilon)

policy = mdp.extractPolicy(V)
print("Policy: ")
print(policy)

V = mdp.evaluatePolicy(policy)
print("Policy Evaluation: ")
print(V)

[policy,V,iterId] = mdp.policyIteration(np.array([0,0,0,0]))
print("Policy Iteration:")
print([policy,V,iterId])

[V,iterId,epsilon] = mdp.evaluatePolicyPartially(np.array([1,0,1,0]),np.array([0,10,0,13]))
print("Partial Policy Evaluation:")
print([V, iterId, epsilon])

[policy,V,iterId,tolerance] = mdp.modifiedPolicyIteration(np.array([1,0,1,0]),np.array([0,10,0,13]))
print("Modified Policy Iteration:")
print([policy,V,iterId,tolerance])