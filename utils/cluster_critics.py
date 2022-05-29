import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain

class ClusterCritic(nn.Module):
    """
    Attention network, used as critic for all CLUSTERS.
    Each CLUSTER gets its own observation and action, and can also 
    attend over the other CLUSTERS' encoded observations and actions.
    """
    def __init__(self, sa_sizes, cluster_list, n_clusters=5,
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
        super(ClusterCritic, self).__init__()
        assert (hidden_dim % attend_heads) == 0
        self.sa_sizes = sa_sizes
        self.nagents = len(sa_sizes)
        self.attend_heads = attend_heads

        self.n_clusters = n_clusters                # number of clusters (fixed)
        self.cluster_list = cluster_list            # dictionary of clusters (e.g., {0: [1, 2, 3], 1: [4, 5]})

        self.c_critic_encoders = nn.ModuleList()    # encoders for state-action of each cluster
        self.c_critics = nn.ModuleList()
        self.c_state_encoders = nn.ModuleList()     # encoders for states of each cluster

        '''
        iterate over clusters (05/23, yuseung)
        '''
        # assert self.nagents % 5 == 0, 'nagents should be a multiple of 5'

        for clst in self.cluster_list.values():
            # print(sa_sizes(clst[0]))
            sdim, adim = sa_sizes[clst[0]]            # WARNING: sdim, adim of each agent in a cluster should be equal
            
            # extend dim by the number of agents
            c_sdim = sdim * len(clst)
            c_adim = adim * len(clst)
            c_idim = c_sdim + c_adim
            c_odim = c_adim

            c_critic_encoder = nn.Sequential()
            if norm_in:
                c_critic_encoder.add_module('c_enc_bn', nn.BatchNorm1d(c_idim, affine=False))
            c_critic_encoder.add_module('c_enc_fc1', nn.Linear(c_idim, hidden_dim))
            c_critic_encoder.add_module('c_enc_nl', nn.LeakyReLU())
            self.c_critic_encoders.append(c_critic_encoder)

            c_critic = nn.Sequential()
            c_critic.add_module('c_critic_fc1', nn.Linear(2 * hidden_dim, hidden_dim))
            c_critic.add_module('c_critic_nl', nn.LeakyReLU())
            c_critic.add_module('c_critic_fc2', nn.Linear(hidden_dim, c_odim))
            self.c_critics.append(c_critic)

            c_state_encoder = nn.Sequential()
            if norm_in:
                c_state_encoder.add_module('cs_enc_bn', nn.BatchNorm1d(c_sdim, affine=False))
            c_state_encoder.add_module('cs_enc_fc1', nn.Linear(c_sdim, hidden_dim))
            c_state_encoder.add_module('cs_enc_nl', nn.LeakyReLU())
            self.c_state_encoders.append(c_state_encoder)

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
                               self.c_value_extractors, self.c_critic_encoders]

    def shared_parameters(self):
        """
        Parameters shared across agents and reward heads
        """
        return chain(*[m.parameters() for m in self.shared_modules])

    def scale_shared_grads(self):
        """
        Scale gradients for parameters that are shared since they accumulate
        gradients from the critic loss function multiple times
        """
        for p in self.shared_parameters():
            p.grad.data.mul_(1. / self.n_clusters)

    def calculateAttention(self, cluster_list, queryHeads, keyHeads, valueHeads):
        clst_attention_values = [[] for _ in range(len(cluster_list))]
        clst_attend_logits = [[] for _ in range(len(cluster_list))]
        clst_attend_probs = [[] for _ in range(len(cluster_list))]

        # calculate attention per head
        for curr_head_keys, curr_head_values, curr_head_selectors in zip(keyHeads, valueHeads, queryHeads):
            # iterate over clusters
            for i, c_i, selector in zip(range(len(cluster_list)), cluster_list.keys(), curr_head_selectors):
                keys = [k for j, k in enumerate(curr_head_keys) if j != c_i]
                values = [v for j, v in enumerate(curr_head_values) if j != c_i]

                # calculate attention across clusters
                attend_logits = torch.matmul(
                                        selector.view(selector.shape[0], 1, -1),
                                        torch.stack(keys).permute(1, 2, 0)
                                        )

                # scale dot-products by size of key (from Attention is All You Need)
                scaled_attend_logits = attend_logits / np.sqrt(keys[0].shape[1])

                attend_weights = F.softmax(scaled_attend_logits, dim=2)

                attention_values = (torch.stack(values).permute(1, 2, 0) *
                                        attend_weights).sum(dim=2)

                clst_attention_values[i].append(attention_values)
                clst_attend_logits[i].append(attend_logits)
                clst_attend_probs[i].append(attend_weights)

        return clst_attention_values, clst_attend_logits, clst_attend_probs

    def forward(self, inps, return_q=True, return_all_q=False,
                regularize=False, return_attend=False, logger=None, niter=0):

        # state, actions of each cluster
        states = [s for s, a in inps]
        actions = [a for s, a in inps]

        # contatenate the inputs in the same cluster
        clst_states = []
        clst_actions = []
        c_inps = []

        for clst_idx, agents in self.cluster_list.items():
            c_states = [states[i] for i in agents]
            c_actions = [actions[i] for i in agents]

            clst_state = torch.cat(c_states, dim=1)
            clst_action = torch.cat(c_actions, dim=1)

            clst_states.append(clst_state)
            clst_actions.append(clst_action)

            c_inps.append(torch.cat((clst_state, clst_action), dim=1))

        '''encode (states, actions) of each agent'''
        # extract state-action encoding for each agent
        sa_encodings = [encoder(inp) for encoder, inp in zip(self.c_critic_encoders, c_inps)]
        # extract state encoding for each agent that we're returning Q for
        s_encodings = [self.c_state_encoders[c_i](clst_states[c_i]) for c_i in self.cluster_list.keys()]
        # extract keys for each head for each agent
        clst_head_keys = [[k_ext(enc) for enc in sa_encodings] for k_ext in self.c_key_extractors]
        # extract sa values for each head for each agent
        clst_head_values = [[v_ext(enc) for enc in sa_encodings] for v_ext in self.c_value_extractors]
        # extract selectors for each head for each agent that we're returning Q for
        clst_head_selectors = [[sel_ext(enc) for i, enc in enumerate(s_encodings) if i in self.cluster_list.keys()]
                              for sel_ext in self.c_selector_extractors]

        return self.calculateAttention(self.cluster_list, 
                                        clst_head_selectors, 
                                        clst_head_keys, 
                                        clst_head_values)