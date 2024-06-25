import torch
from torch import nn
import numpy as np
import pennylane as qml

from quantum_circuits import *
from utils_eqgnn import *

"""
    Quantum Lie-Equivariant Block (QLieGEB).
    
        - Given the Lie generators found (i.e.: through LieGAN, oracle-preserving latent flow, or some other approach
          that we develop further), once the metric tensor J is found via the equation:

                          L.J + J.(L^T) = 0,
                          
          we just have to specify the metric to make the model symmetry-preserving to the corresponding Lie group. 
          In the cells below, I will show first how the model preserves symmetries (starting with the default Lorentz group),
          and when we change J to some other metric (Euclidean, for example), Lorentz boosts break equivariance, while other
          transformations preserve it (rotations, for the example shown in the cells below)
"""
class QLieGEB(nn.Module):
    def __init__(self, n_input, n_output, n_hidden, n_node_attr=0,
                 dropout = 0., c_weight=1.0, last_layer=False, A=None, include_x=False, 
                 model_type='classical'):
        
        super(QLieGEB, self).__init__()
        self.c_weight = c_weight
        n_edge_attr = 2 if not include_x else 10 # dims for Minkowski norm & inner product

        self.include_x = include_x

        """
            phi_e: input size: n_qubits -> output size: n_qubits
            n_hidden has to be equal to n_input (n_input * 2 + n_edge_attr),
            but this is just considering that this is a simple working example.
        """

        
        n_hidden = n_input * 2 + n_edge_attr
        
        if model_type in ['phi_e', 'quantum']:
            self.phi_e = DressedQuantumNet(n_qubits=n_input * 2 + n_edge_attr, q_depth=2)
        else:
            self.phi_e = nn.Sequential(
                        nn.Linear(n_input * 2 + n_edge_attr, n_hidden, bias=False),
                        nn.BatchNorm1d(n_hidden),
                        nn.ReLU(),
                        nn.Linear(n_hidden, n_hidden),
                        nn.ReLU())
            

        if model_type in ['phi_h', 'quantum']:
            self.phi_h = nn.Sequential(
                        nn.Linear(n_hidden + n_input + n_node_attr, n_hidden, bias=False),
                        nn.BatchNorm1d(n_hidden),
                        nn.ReLU(),
                        DressedQuantumNet(n_qubits=n_hidden, q_depth=2),
                        nn.Linear( n_hidden, n_output, bias=False),)
        else:
            self.phi_h = nn.Sequential(
                nn.Linear(n_hidden + n_input + n_node_attr, n_hidden),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_output))

        

        if model_type in ['phi_x', 'quantum']:
            down_projection = nn.Linear(n_hidden, 1, bias=False)
            torch.nn.init.xavier_uniform_(down_projection.weight, gain=0.001)
            self.phi_x = nn.Sequential(
                            DressedQuantumNet(n_qubits=n_hidden, q_depth=2),
                            down_projection)
                        
        else:
            layer = nn.Linear(n_hidden, 1, bias=False)
            torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
            self.phi_x = nn.Sequential(
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                layer)

        
        if model_type in ['phi_m', 'quantum']:
            self.phi_m = nn.Sequential(nn.Linear(n_hidden, 1),
                                       DressedQuantumNet(n_qubits=1, q_depth=1))
        else:
            self.phi_m = nn.Sequential(
                nn.Linear(n_hidden, 1),
                nn.Sigmoid())    

        
        self.last_layer = last_layer
        if last_layer:
            del self.phi_x

        self.A = A
        self.norm_fn = normA_fn(A) if A is not None else normsq4
        self.dot_fn = dotA_fn(A) if A is not None else dotsq4

    def m_model(self, hi, hj, norms, dots):
        out = torch.cat([hi, hj, norms, dots], dim=1)
        # print("Before embedding to |phi_e> : ", out, out.shape)
        out = self.phi_e(out).squeeze(0)
        # print("Input of phi_m (out):", out, out.shape)
        w = self.phi_m(out)
        out = out * w
        return out

    def m_model_extended(self, hi, hj, norms, dots, xi, xj):
        out = torch.cat([hi, hj, norms, dots, xi, xj], dim=1)
        out = self.phi_e(out).squeeze(0)
        w = self.phi_m(out)
        out = out * w
        return out

    def h_model(self, h, edges, m, node_attr):
        i, j = edges
        agg = unsorted_segment_sum(m, i, num_segments=h.size(0))
        agg = torch.cat([h, agg, node_attr], dim=1)
        # print("h shape: {}, agg shape: {}, phi_h(agg).shape = {}".format(h.shape, agg.shape, self.phi_h(agg).shape))
        out = h + self.phi_h(agg)
        return out

    def x_model(self, x, edges, x_diff, m):
        i, j = edges
        trans = x_diff * self.phi_x(m)
        # From https://github.com/vgsatorras/egnn
        # This is never activated but just in case it explosed it may save the train
        # From https://github.com/vgsatorras/egnn
        # This is never activated but just in case it explosed it may save the train
        trans = torch.clamp(trans, min=-100, max=100)
        agg = unsorted_segment_mean(trans, i, num_segments=x.size(0))
        x = x + agg * self.c_weight
        return x

    def minkowski_feats(self, edges, x):
        i, j = edges
        x_diff = x[i] - x[j]
        norms = self.norm_fn(x_diff).unsqueeze(1)
        dots = self.dot_fn(x[i], x[j]).unsqueeze(1)
        norms, dots = psi(norms), psi(dots)
        return norms, dots, x_diff

    def forward(self, h, x, edges, node_attr=None):
        i, j = edges
        norms, dots, x_diff = self.minkowski_feats(edges, x)

        if self.include_x:
            m = self.m_model_extended(h[i], h[j], norms, dots, x[i], x[j])
        else:
            # try:
            # print(h.shape, i, j)
            m = self.m_model(h[i], h[j], norms, dots) # [B*N, hidden]
        if not self.last_layer:
            x = self.x_model(x, edges, x_diff, m)
        h = self.h_model(h, edges, m, node_attr)
        return h, x, m

class LieEQGNN(nn.Module):
    r''' Implementation of Lie-Equivariant Quantum Graph Neural Network (Lie-EQGNN).

    Args:
        - `n_scalar` (int): number of input scalars.
        - `n_hidden` (int): dimension of latent space.
        - `n_class`  (int): number of output classes.
        - `n_layers` (int): number of QLieGEB layers.
        - `c_weight` (float): weight c in the x_model.
        - `dropout`  (float): dropout rate.
    '''
    def __init__(self, n_scalar, n_hidden, n_class = 2, n_layers = 6, c_weight = 1e-3, dropout = 0., A=None, include_x=False,
                model_type='classical'):
        
        super(LieEQGNN, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.embedding = nn.Linear(n_scalar, n_hidden)
        self.QLieGEBs = nn.ModuleList([QLieGEB(self.n_hidden, self.n_hidden, self.n_hidden, 
                                    n_node_attr=n_scalar, dropout=dropout,
                                    c_weight=c_weight, last_layer=(i==n_layers-1), A=A, include_x=include_x,
                                    model_type=model_type)
                                    for i in range(n_layers)])
        
        self.graph_dec = nn.Sequential(nn.Linear(self.n_hidden, self.n_hidden),
                                       nn.ReLU(),
                                       nn.Dropout(dropout),
                                       nn.Linear(self.n_hidden, n_class)) # classification

    def forward(self, scalars, x, edges, node_mask, edge_mask, n_nodes):
        h = self.embedding(scalars)
        
        # print("h before (just the first particle): \n", h[0].cpu().detach().numpy())
        for i in range(self.n_layers):
            h, x, _ = self.QLieGEBs[i](h, x, edges, node_attr=scalars)
        
        # print("h after (just the first particle): \n", h[0].cpu().detach().numpy())
        
        h = h * node_mask
        h = h.view(-1, n_nodes, self.n_hidden)
        h = torch.mean(h, dim=1)
        pred = self.graph_dec(h)
        return pred.squeeze(1)