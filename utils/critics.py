from time import clock_settime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain

from utils.clustering import cluster_agents
# from .critic_buffer import CriticBuffer
from .cluster_critics import ClusterCritic

cluster_critic_lr = 0.001

class AttentionCritic(nn.Module):
    """
    Attention network, used as critic for all agents. Each agent gets its own
    observation and action, and can also attend over the other agents' encoded
    observations and actions.
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
        super(AttentionCritic, self).__init__()
        assert (hidden_dim % attend_heads) == 0
        self.sa_sizes = sa_sizes
        self.nagents = len(sa_sizes)
        self.attend_heads = attend_heads

        self.critic_encoders = nn.ModuleList()
        self.critics = nn.ModuleList()
        self.state_encoders = nn.ModuleList()

        '''add self.critic_buffer (yuseung, 05/20)'''
        # self.critic_buffer = CriticBuffer(attend_heads=attend_heads)

        # iterate over agents
        for sdim, adim in sa_sizes:
            idim = sdim + adim
            odim = adim
            encoder = nn.Sequential()
            if norm_in:
                encoder.add_module('enc_bn', nn.BatchNorm1d(idim, affine=False))
            encoder.add_module('enc_fc1', nn.Linear(idim, hidden_dim))
            encoder.add_module('enc_nl', nn.LeakyReLU())
            self.critic_encoders.append(encoder)

            critic = nn.Sequential()
            critic.add_module('critic_fc1', nn.Linear(2 * hidden_dim, hidden_dim))
            critic.add_module('critic_nl', nn.LeakyReLU())
            critic.add_module('critic_fc2', nn.Linear(hidden_dim, odim))
            self.critics.append(critic)

            state_encoder = nn.Sequential()
            if norm_in:
                state_encoder.add_module('s_enc_bn', nn.BatchNorm1d(sdim, affine=False))
            state_encoder.add_module('s_enc_fc1', nn.Linear(sdim, hidden_dim))
            state_encoder.add_module('s_enc_nl', nn.LeakyReLU())
            self.state_encoders.append(state_encoder)

        attend_dim = hidden_dim // attend_heads
        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()
        self.value_extractors = nn.ModuleList()

        for i in range(attend_heads):
            self.key_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.selector_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.value_extractors.append(nn.Sequential(nn.Linear(hidden_dim, attend_dim), nn.LeakyReLU()))

        self.shared_modules = [self.key_extractors, self.selector_extractors,
                               self.value_extractors, self.critic_encoders]

        '''Init ClusterCritic for cluster attention (05/27 Yuseung)'''
        self.n_clusters = n_clusters

        '''TODO: implememt clustering (05/28)'''
        self.cluster_list = {0: [0, 1, 2, 3, 4],
                            1: [5, 6, 7, 8, 9],
                            2: [10, 11, 12, 13, 14]}

        self.cluster_critic = ClusterCritic(sa_sizes=self.sa_sizes,
                                            cluster_list=self.cluster_list,
                                            n_clusters=self.n_clusters,
                                            hidden_dim=32,
                                            attend_heads=1)

        self.cluster_critic_optimizer = torch.optim.Adam(self.cluster_critic.parameters(),
                                                        lr=cluster_critic_lr,
                                                        weight_decay=1e-3)

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
            p.grad.data.mul_(1. / self.nagents)

    """
    queryHeads : list of query values for each attention head (list of list of queries)
    """
    def calculateAttention(self, agents, queryHeads, keyHeads, valueHeads):
        all_attention_values = [[] for _ in range(len(agents))]
        all_attend_logits = [[] for _ in range(len(agents))]
        all_attend_probs = [[] for _ in range(len(agents))]

        # calculate attention per head
        for i_head, curr_head_keys, curr_head_values, curr_head_selectors in zip(
                range(len(keyHeads)), keyHeads, valueHeads, queryHeads):

            # iterate over agents
            for i, a_i, selector in zip(range(len(agents)), agents, curr_head_selectors):
                keys = [k for j, k in enumerate(curr_head_keys) if j != a_i]
                values = [v for j, v in enumerate(curr_head_values) if j != a_i]

                '''TODO: implement a pipeline to consider N previous states for attention'''

                # calculate attention across agents
                attend_logits = torch.matmul(selector.view(selector.shape[0], 1, -1),
                                                torch.stack(keys).permute(1, 2, 0))

                # scale dot-products by size of key (from Attention is All You Need)
                scaled_attend_logits = attend_logits / np.sqrt(keys[0].shape[1])

                '''add critic buffer (yuseung, 05/20)'''
                # prev_attend = self.critic_buffer.get_prev_attend(i_head, scaled_attend_logits.detach())
                # if prev_attend is not None:
                #     scaled_attend_logits = 0.2 * prev_attend + 0.8 * scaled_attend_logits

                attend_weights = F.softmax(scaled_attend_logits, dim=2)

                attention_values = (torch.stack(values).permute(1, 2, 0) *
                                attend_weights).sum(dim=2)

                all_attention_values[i].append(attention_values)
                all_attend_logits[i].append(attend_logits)
                all_attend_probs[i].append(attend_weights)
        
        return all_attention_values, all_attend_logits, all_attend_probs

    def forward(self, inps, agents=None, return_q=True, return_all_q=False,
                regularize=False, return_attend=False, logger=None, niter=0):
        """
        Inputs:
            inps (list of PyTorch Matrices): Inputs to each agents' encoder (batch of obs + ac)
            agents (int): indices of agents to return Q for
            return_q (bool): return Q-value
            return_all_q (bool): return Q-value for all actions
            regularize (bool): returns values to add to loss function for
                               regularization
            return_attend (bool): return attention weights per agent
            logger (TensorboardX SummaryWriter): If passed in, important values are logged
        """
        if agents is None:
            agents = range(len(self.critic_encoders))
        self.agents = agents

        # state, actions of each agent
        agent_inps = inps
        states = [s for s, a in inps]
        actions = [a for s, a in inps]
        inps = [torch.cat((s, a), dim=1) for s, a in inps]

        '''step 1) encode (states, actions) of each agent'''

        # extract state-action encoding for each agent
        sa_encodings = [encoder(inp) for encoder, inp in zip(self.critic_encoders, inps)]
        # extract state encoding for each agent that we're returning Q for
        s_encodings = [self.state_encoders[a_i](states[a_i]) for a_i in agents]
        # extract keys for each head for each agent
        all_head_keys = [[k_ext(enc) for enc in sa_encodings] for k_ext in self.key_extractors]
        # extract sa values for each head for each agent
        all_head_values = [[v_ext(enc) for enc in sa_encodings] for v_ext in self.value_extractors]
        # extract selectors for each head for each agent that we're returning Q for
        all_head_selectors = [[sel_ext(enc) for i, enc in enumerate(s_encodings) if i in agents]
                              for sel_ext in self.selector_extractors]

        # calculate agents attention
        agent_attention_values, agent_attend_logits, agent_attend_probs = self.calculateAttention(agents, 
                                                                                            all_head_selectors, 
                                                                                            all_head_keys, 
                                                                                            all_head_values)


        print(f'agent_attention_value: {agent_attention_values[0][0].shape}')
        print(f'agent_attend_logit: {agent_attend_logits[0][0].shape}')
        print(f'agent_attend_probs: {agent_attend_probs[0][0].shape}')

        # calculate cluster attention before calculating Q value (05/27, Yuseung)
        clst_attention_values, clst_attend_logits, clst_attend_probs = self.cluster_critic(agent_inps)

        print(f'clst_attention_values: {clst_attention_values[0][0].shape}')
        print(f'clst_attend_logits: {clst_attend_logits[0][0].shape}')
        print(f'clst_attend_probs: {clst_attend_probs[0][0].shape}')

        ########### TODO: extend the cluster attentions to agent attentions  ###########
        clst_logits_extended = []
        clst_probs_extended = []

        for i_heads in range(self.attend_heads):
            for clst_idx, agents in self.cluster_list.items():
                clst_logit = torch.ones_like(agent_attend_logits[0][0])
                clst_probs = torch.ones_like(agent_attend_probs[0][0])
                
                cnt = 0
                for idx, logit, probs in zip(range(len(clst_attend_logits)), clst_attend_logits, clst_attend_probs):
                    if idx == clst_idx:
                        continue
                    clst_logit[:, :, cnt] = logit
                    clst_probs[:, :, cnt] = probs
                    cnt += 1

                clst_logits_extended.append(clst_logit)
                clst_probs_extended.append(clst_probs)

        print(f'clst_logits_extended[0]: {clst_logits_extended[0].shape}')
        print(f'clst_probs_extended[0]: {clst_probs_extended[0].shape}')

        ########### TODO: extend the cluster attentions to agent attentions  ###########


        '''TODO: add agent_attention_values and clst_attention_values (05/28)'''
        tot_attention_values = []
        tot_attend_logits = []
        tot_attend_probs = []

        for i_heads in range(self.attend_heads):
            for i, a_i in enumerate(self.agents):
                clst_idx = [k for k, v in self.cluster_list.items() if a_i in v][0]

                tot_attention_value = agent_attention_values[i][i_heads] * (1 + clst_attention_values[clst_idx][i_heads])
                tot_attend_logit = agent_attend_logits[i][i_heads] * (1 + clst_attend_logits[clst_idx][i_heads])
                tot_attend_prob = agent_attend_probs[i][i_heads] * (1 + clst_attend_probs[clst_idx][i_heads])

                tot_attention_values[i].append(tot_attention_value)
                tot_attend_logits[i].append(tot_attend_logit)
                tot_attend_probs[i].append(tot_attend_prob)

        # calculate Q per agent (considering clusters)
        all_rets = []

        for i, a_i in enumerate(self.agents):
            head_entropies = [(-((probs + 1e-8).log() * probs).squeeze().sum(1)
                                .mean()) for probs in tot_attend_probs[i]]
            agent_rets = []

            critic_in = torch.cat((s_encodings[i], *tot_attention_values[i]), dim=1)

            all_q = self.critics[a_i](critic_in)
            int_acs = actions[a_i].max(dim=1, keepdim=True)[1]
            q = all_q.gather(1, int_acs)

            if return_q:
                agent_rets.append(q)
            if return_all_q:
                agent_rets.append(all_q)
            if regularize:
                # regularize magnitude of attention logits
                attend_mag_reg = 1e-3 * sum((logit**2).mean() for logit in
                                            tot_attend_logits[i])
                regs = (attend_mag_reg,)
                agent_rets.append(regs)

            if return_attend:
                agent_rets.append(np.array(tot_attend_probs[i]))
            if logger is not None:
                logger.add_scalars('agent%i/attention' % a_i,
                                    dict(('head%i_entropy' % h_i, ent) for h_i, ent
                                        in enumerate(head_entropies)),
                                    niter)
            if len(agent_rets) == 1:
                all_rets.append(agent_rets[0])
            else:
                all_rets.append(agent_rets)
        if len(all_rets) == 1:
            return all_rets[0]
        else:
            return all_rets



"""
(5/24) Attention probability is important than the value itself? How one cluster should pay attention to the other group..
ex) cluster A has high attention probability toward cluster B
-> amplify attention values of agents in A toward other agents in B, by multiplying certain factor 
"""