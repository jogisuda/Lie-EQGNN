import numpy as np
import torch
import pennylane as qml
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import to_dense_adj


n_qubits = 10
device='cpu'

dev = qml.device('default.qubit', wires=n_qubits)


def H_layer(nqubits):
    """Layer of single-qubit Hadamard gates.
    """
    for idx in range(nqubits):
        qml.Hadamard(wires=idx)


def RY_layer(w):
    """Layer of parametrized qubit rotations around the y axis.
    """
    for idx, element in enumerate(w):
        qml.RY(element, wires=idx)


def entangling_layer(nqubits):
    """Layer of CNOTs followed by another shifted layer of CNOT.
    """
    # In other words it should apply something like :
    # CNOT  CNOT  CNOT  CNOT...  CNOT
    #   CNOT  CNOT  CNOT...  CNOT
    for i in range(0, nqubits - 1, 2):  # Loop over even indices: i=0,2,...N-2
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, nqubits - 1, 2):  # Loop over odd indices:  i=1,3,...N-3
        qml.CNOT(wires=[i, i + 1])


@qml.qnode(dev, interface="torch")
def quantum_net(q_input_features, q_weights_flat, q_depth, n_qubits, measure_only=n_qubits):
    """
    The variational quantum circuit.
    """

    # Reshape weights
    q_weights = q_weights_flat.reshape(q_depth, n_qubits)

    # Start from state |+> , unbiased w.r.t. |0> and |1>
    H_layer(n_qubits)

    # Embed features in the quantum node
    RY_layer(q_input_features)

    # Sequence of trainable variational layers
    for k in range(q_depth):
        entangling_layer(n_qubits)
        RY_layer(q_weights[k])

    # Expectation values in the Z basis
    exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(n_qubits)]

    # if n_qubits == 1:
    #     print("1 only: ", tuple(exp_vals))
    # elif n_qubits == 10:
    #     print("10 only: ", tuple(exp_vals))
    # return tuple(exp_vals)

    return tuple(exp_vals)


class DressedQuantumNet(nn.Module):
    """
    Torch module implementing the *dressed* quantum net.
    """

    def __init__(self, n_qubits, q_depth = 1, q_delta=0.001):
        """
        Definition of the *dressed* layout.
        """

        super().__init__()
        self.n_qubits = n_qubits
        self.q_depth = q_depth
        self.q_params = nn.Parameter(q_delta * torch.randn(q_depth * n_qubits))

    def forward(self, input_features):
        """
        Defining how tensors are supposed to move through the *dressed* quantum
        net.
        """

        # Quantum Embedding (U(X))
        q_in = torch.tanh(input_features) * np.pi / 2.0

        # Apply the quantum circuit to each element of the batch and append to q_out
        q_out = torch.Tensor(0, self.n_qubits)
        q_out = q_out.to(device)

        # print(q_in, q_in.shape, self.n_qubits)
        # for batch in q_in:
        for elem in q_in:
            # print(quantum_net(elem, self.q_params, self.q_depth, self.n_qubits))
            # print(quantum_net(elem, self.q_params, self.q_depth, self.n_qubits))
            # if q_in.shape[1] == 10:
            #     print("10 works: ", quantum_net(elem, self.q_params, self.q_depth, self.n_qubits))
            if q_in.shape[1] == 1:
                # print("1 DOES NOT WORK: ", tuple([quantum_net(elem, self.q_params, self.q_depth, self.n_qubits)] ) )
                q_out_elem = torch.hstack(tuple([quantum_net(elem, self.q_params, self.q_depth, self.n_qubits)])).float().unsqueeze(0)
            else:
                q_out_elem = torch.hstack(quantum_net(elem, self.q_params, self.q_depth, self.n_qubits)).float().unsqueeze(0)
            q_out = torch.cat((q_out, q_out_elem))

        # return the batch measurement of the PQC
        return q_out#.unsqueeze(0)