import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain

from .critics import AttentionCritic

class ClusterCritic(AttentionCritic):
    """
    Attention network, used as critic for all CLUSTERS.
    Each CLUSTER gets its own observation and action, and can also 
    attend over the other CLUSTERS' encoded observations and actions.
    """
    def __init__(self, sa_sizes, n_clusters=5,
                    hidden_dim=32, norm_in=True, attend_heads=1):
        """
        Inputs:
            sa_sizes (list of (int, int)): Size of state and action spaces per
                                          agent
            hidden_dim (int): Number of hidden dimensions
            norm_in (bool): Whether to apply BatchNorm to input
            attend_heads (int): Number of attention heads to use (use a number
                                that hidden_dim is divisible by)
        """
        # super(ClusterCritic, self).__init__()
        assert (hidden_dim % attend_heads) == 0
        self.sa_sizes = sa_sizes
        self.nagents = len(sa_sizes)
        self.attend_heads = attend_heads

        self.n_clusters = n_clusters                # number of clusters (fixed)
        # self.n_clusters = self.nagents // 5 

        self.clst_encoders = nn.ModuleList()        # encoders for state-action of each cluster
        self.clst_critics = nn.ModuleList()
        self.clst_state_encoders = nn.ModuleList()  # encoders for states of each cluster

        '''
        iterate over clusters (05/23, yuseung)
        each cluster has exactly 5 agents (assumption)
        '''
        assert self.nagents % 5 == 0, 'nagents should be a multiple of 5'

        for n in range(0, self.nagents, 5):
            # sdim, adim of each agent in a cluster should be equal
            sdim, adim = sa_sizes[n]
            
            idim = sdim + adim
            odim = adim
            clst_encoder = nn.Sequential()
            if norm_in:
                clst_encoder.add_module('c_enc_bn', nn.BatchNorm1d(idim, affine=False))
            clst_encoder.add_module('c_enc_fc1', nn.Linear(idim, hidden_dim))
            clst_encoder.add_module('c_enc_nl', nn.LeakyReLU())
            self.clst_encoders.append(clst_encoder)

            clst_critic = nn.Sequential()
            clst_critic.add_module('c_critic_fc1', nn.Linear(2 * hidden_dim, hidden_dim))
            clst_critic.add_module('c_critic_nl', nn.LeakyReLU())
            clst_critic.add_module('c_critic_fc2', nn.Linear(hidden_dim, odim))
            self.clst_critics.append(clst_critic)

            clst_state_encoder = nn.Sequential()
            if norm_in:
                clst_state_encoder.add_module('cs_enc_bn', nn.BatchNorm1d(sdim, affine=False))
            clst_state_encoder.add_module('cs_enc_fc1', nn.Linear(sdim, hidden_dim))
            clst_state_encoder.add_module('cs_enc_nl', nn.LeakyReLU())
            self.clst_state_encoders.append(clst_state_encoder)

        attend_dim = hidden_dim // attend_heads

        # (5/24) Query, key, value extractors for clusters
        self.c_key_extractors = nn.ModuleList()
        self.c_selector_extractors = nn.ModuleList()
        self.c_value_extractors = nn.ModuleList()

        for i in range(attend_heads):
             # (5/24) For clusters
            self.c_key_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.c_selector_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.c_value_extractors.append(nn.Sequential(nn.Linear(hidden_dim, attend_dim), nn.LeakyReLU()))

        self.shared_modules = [self.c_key_extractors, self.c_selector_extractors,
                               self.c_value_extractors, self.clst_encoders]

    def shared_parameters(self):
        pass

    def scale_shared_grads(self):
        pass

    def calculateAttention(self, clusters, queryHeads, keyHeads, valueHeads):
        return super().calculateAttention(clusters, queryHeads, keyHeads, valueHeads)

    def forward(self, inps, agents=None, return_q=True, return_all_q=False,
                regularize=False, return_attend=False, logger=None, niter=0):
        pass

