# General imports
import numpy as np
import matplotlib.pyplot as plt

# Qiskit Circuit imports
from qiskit.circuit import QuantumCircuit, QuantumRegister, Parameter, ParameterVector, ParameterExpression
from qiskit.circuit.library import TwoLocal

# Qiskit imports
import qiskit as qk
from qiskit.utils import QuantumInstance
from qiskit import Aer
from qiskit.tools.visualization import circuit_drawer
from qiskit.quantum_info import state_fidelity
from qiskit import BasicAer
from qiskit.providers.aer.noise import NoiseModel, amplitude_damping_error


# Qiskit Machine Learning imports
import qiskit_machine_learning as qkml
from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit_machine_learning.connectors import TorchConnector

# PyTorch imports
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F


from math import pi
from math import *




def encoding_circuit(inputs, num_qubits = 1, *args):
    """
    Encode classical input data (i.e. the state of the enironment) on a quantum circuit. 
    To be used inside the `parametrized_circuit` function. 
    
    Args
    -------
    inputs (list): a list containing the classical inputs.
    num_qubits (int): number of qubits in the quantum circuit.
    
    Return
    -------
    qc (QuantumCircuit): quantum circuit with encoding gates.
    
    """
    
    qc = qk.QuantumCircuit(num_qubits)
    
    # Encode data with a RX rotation
    for i in range(num_qubits): 
        qc.u(inputs[i*2],inputs[i*2+1],0,i)
        
    return qc

def action_circuit(num_qubits = 1,*args):
    ac = qk.QuantumRegister(num_qubits)
    qc = qk.QuantumCircuit(ac)
    input = qk.circuit.ParameterVector('x', 4*num_qubits)
    for i in range(num_qubits): 
        qc.rx(input[4*i], i)
        qc.rz(input[4*i+1], i)
        qc.u(input[4*i+2],input[4*i+3],0,i)
    return qc



def parametrized_circuit(num_qubits = 1, reps = 1, insert_barriers = True, ):
    """
    Create the Parameterized Quantum Circuit (PQC) for estimating Q-values.
    It implements the architecure proposed in Skolik et al. arXiv:2104.15084.
    
    Args
    -------
    num_qubit (int): number of qubits in the quantum circuit. 

    reps (int): number of repetitions (layers) in the variational circuit. 
    insert_barrirerd (bool): True to add barriers in between gates, for better drawing of the circuit. 

    
    Return
    -------
    qc (QuantumCircuit): the full parametrized quantum circuit. 
    """
    
    qr = qk.QuantumRegister(num_qubits)
    qc = qk.QuantumCircuit(qr)
    
    

        
    # Define a vector containg Inputs as parameters (*not* to be optimized)
    inputs = qk.circuit.ParameterVector('x', 2*num_qubits)
            
    # Define a vector containng variational parameters
    θ = qk.circuit.ParameterVector('θ', 3 * num_qubits * reps)
    qc.compose(encoding_circuit(inputs, num_qubits = num_qubits), inplace = True)
    if insert_barriers: qc.barrier()
    
    # Iterate for a number of repetitions
    for rep in range(reps):

        # Encode classical input data

            
        # Variational circuit (does the same as TwoLocal from Qiskit)
        for qubit in range(num_qubits):
            qc.rx(θ[qubit + 3*num_qubits*(rep)], qubit)
            qc.rz(θ[qubit + 3*num_qubits*(rep) + num_qubits], qubit)
            qc.rx(θ[qubit + 3*num_qubits*(rep) + 2*num_qubits], qubit)
        if insert_barriers: qc.barrier()
            
        # Add entanglers (this code is for a circular entangler)
        if num_qubits>2:
            qc.cnot(qr[-1], qr[0])
            for qubit in range(num_qubits-1):
                qc.cnot(qr[qubit], qr[qubit+1])
            if insert_barriers: qc.barrier()
        elif num_qubits==2:
            qc.cnot(qr[-1], qr[0])
        
                    

        
    return qc



# Select the number of qubits
num_qubits = 1

# Generate the Parametrized Quantum Circuit (note the flags reuploading and reps)
policy_qc = parametrized_circuit(num_qubits = num_qubits, 
                          reps = 2)

value_qc=parametrized_circuit(num_qubits = 2*num_qubits, 
                          reps = 2)

# Fetch the parameters from the circuit and divide them in Inputs (X) and Trainable Parameters (params)
# The first four parameters are for the inputs 
policy_X = list(policy_qc.parameters)[: 2*num_qubits]
value_X=list(value_qc.parameters)[: 4*num_qubits]

# The remaining ones are the trainable weights of the quantum neural network
policy_params = list(policy_qc.parameters)[num_qubits:]
value_params=list(value_qc.parameters)[2*num_qubits:]

action_qc=action_circuit(num_qubits=num_qubits)
action_X=list(action_qc.parameters)



# Select a quantum backend to run the simulation of the quantum circuit
qi = QuantumInstance(qk.BasicAer.get_backend('statevector_simulator'))

# Create a Quantum Neural Network object starting from the quantum circuit defined above
policy_qnn = CircuitQNN(policy_qc, input_params=policy_X, weight_params=policy_params, 
                 quantum_instance = qi)

value_qnn = CircuitQNN(value_qc, input_params=value_X, weight_params=value_params, 
                 quantum_instance = qi)

action_qnn= CircuitQNN(action_qc, input_params=action_X,quantum_instance = qi)


policy_initial = 0.1*(2*torch.rand(policy_qnn.num_weights,requires_grad=True) - 1)
policy_nn = TorchConnector(policy_qnn, policy_initial)

value_initial = 0.1*(2*torch.rand(value_qnn.num_weights,requires_grad=True) - 1)
value_nn = TorchConnector(value_qnn, value_initial)

action_nn=TorchConnector(action_qnn)



class Replay_buffer():
    def __init__(self ,max_size):
        self.storage = []
        self.max_size = max_size
        self.ptr=0
        
    def push(self,data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)
    
    def sample(self, batch_size):
        ind=np.random.randint(0,len(self.storage), size=batch_size)
        k=0
        for i in ind:
            S, S_next, A, R, D =self.storage[i]
            if k==0:
                s=S
                s_next=S_next
                a=A
                r=R
                d=D
                k+=1
            else:

                s=torch.vstack((s,S))
                s_next=torch.vstack((s_next,S_next))
                a=torch.vstack((a,A))
                r=torch.vstack((r,R))
                d=torch.vstack((d,D))
                
            
            
        
        return  s,s_next,a,r,d

def getangle(s):
    sizes=torch.abs(s)
    return torch.tensor([2*torch.acos(sizes[0].type(torch.cfloat)),torch.atan2(torch.imag(s[1].type(torch.cfloat)),torch.real(s[1].type(torch.cfloat)))-torch.atan2(torch.imag(s[0].type(torch.cfloat)),torch.real(s[0].type(torch.cfloat)))])
    
    
class diffQ(nn.Module):
    def __init__(self,valuef,policyf):
        super().__init__()
        self.value=valuef
        self.policy=policyf
        
    def forward(self,s,a=None, currentq=True):
        if currentq==True:
            return self.value(torch.concat((getangle(s).type(torch.float),getangle(a).type(torch.float))))
        else:
            return self.value(torch.concat((getangle(s).type(torch.float), getangle(self.policy(getangle(s).type(torch.float))).type(torch.float))))
    
    
class DDPG(object):
    def __init__(self,max_size=100,learning_rate=1e-3,batch_size=20,gamma=0.997,update_iteration=10,theta=1, tau=1e-3,noise=2*1e-2):
        self.policy=policy_nn
        self.policy_target=policy_nn
        self.value=value_nn
        self.value_target=value_nn
        self.replay_buffer=Replay_buffer(max_size=max_size)
        self.action_operator=action_nn
        self.model= diffQ(self.value, self.policy)
        self.model.train()
        self.opt=optim.Adagrad(self.model.parameters(),lr=learning_rate)
        self.inital=False
        self.batch_size=batch_size
        self.gamma=gamma
        self.update_iteration=update_iteration
        self.theta=theta
        self.tau=tau
        self.noise=noise

    
    def update(self):
        for it in range(self.update_iteration):
            s, s_next, a, r, d= self.replay_buffer.sample(batch_size=self.batch_size)
            a = (a + torch.normal(0, self.noise, size= a.shape))
            target_Q=[]
            current_Q=[]
            value_max=[]
            for i in range(self.batch_size):
                target_Q.append(self.value_target(torch.concat((getangle(s_next[i]).type(torch.float), getangle(self.policy_target(getangle(s_next[i]).type(torch.float))).type(torch.float))))[0])
                current_Q.append(self.model(s[i],a[i],currentq=True)[0])
                value_max.append(self.model(s[i],currentq=False)[0])
            target_Q=torch.tensor(target_Q)
            current_Q=torch.tensor(current_Q)
            current_Q.requires_grad=True
            value_max=torch.tensor(value_max)
            target_Q = r.type(torch.float).squeeze()+ self.gamma*target_Q
            mse_Q = F.mse_loss(current_Q, target_Q)
            value_max = - torch.mean(value_max)
            value_max.requires_grad=True
            self.opt.zero_grad()
            mse_Q.backward()
            self.opt.step()
            self.opt.zero_grad()
            value_max.backward()
            self.opt.step()
            
            for param, target_param in zip(self.policy.parameters(), self.policy_target.parameters()):
                target_param.data.copy_(self.tau*param.data+(1-self.tau)*target_param)
            for param, target_param in zip(self.value.parameters(), self.value_target.parameters()):
                target_param.data.copy_(self.tau*param.data+(1-self.tau)*target_param)
                
    def onetime(self,s,t):
        d=torch.zeros((1,))
        a=self.policy(getangle(s).type(torch.float))
        actioninput=torch.concat((getangle(s).type(torch.float),getangle(a).type(torch.float)))
        s_next=self.action_operator(actioninput)
        angle=getangle(s)
        angle_next=getangle(s_next)
        
        consume=F.relu((torch.tan(angle[0]-pi/2)-torch.tan(angle_next[0]-pi/2)).type(torch.float))+torch.tan(angle_next[1]/4)
        
        labor=-(torch.tan((angle[0]-pi/2))-torch.tan((angle_next[0]-pi/2)))+consume
        
        if t>10:
            labor=labor*2
        r=torch.log(1e-30+consume.type(torch.float))-(self.theta*labor**2)/2    
        if t==19:
            if torch.abs(angle_next[0])**2>0.0001:
                r=torch.tensor([-100])
                d=d+1
            return [s,s_next,a,r,d]
        else:
            return [s,s_next,a,r,d]
        
    def push_buffer(self,s0):
        s=s0
        reward=0
        for t in range(20):
            data=self.onetime(s,t)
            self.replay_buffer.push(data)
            s=data[1]
            reward=reward+data[3]
        print(reward)
        return reward
            
    
        
a=DDPG()
reward=[]
for i in range(5):
    reward.append(a.push_buffer(torch.tensor([1/sqrt(2),1/sqrt(2)])))
for epoch in range(100):
    a.update()
    reward.append(a.push_buffer(torch.tensor([1/sqrt(2),1/sqrt(2)])))       
        
