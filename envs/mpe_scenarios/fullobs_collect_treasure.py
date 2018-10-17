import numpy as np
import seaborn as sns
from multiagent.core import World, Agent, Landmark, Wall
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.cache_dists = True
        world.dim_c = 2
        num_agents = 8
        num_collectors = 6
        num_deposits = num_agents - num_collectors
        world.treasure_types = list(range(num_deposits))
        world.treasure_colors = np.array(
            sns.color_palette(n_colors=num_deposits))
        num_treasures = num_collectors
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.i = i
            agent.name = 'agent %d' % i
            agent.collector = True if i < num_collectors else False
            if agent.collector:
                agent.color = np.array([0.85, 0.85, 0.85])
            else:
                agent.d_i = i - num_collectors
                agent.color = world.treasure_colors[agent.d_i] * 0.35
            agent.collide = True
            agent.silent = True
            agent.ghost = True
            agent.holding = None
            agent.size = 0.05 if agent.collector else 0.075
            agent.accel = 1.5
            agent.initial_mass = 1.0 if agent.collector else 2.25
            agent.max_speed = 1.0
        # add treasures
        world.landmarks = [Landmark() for i in range(num_treasures)]
        for i, landmark in enumerate(world.landmarks):
            landmark.i = i + num_agents
            landmark.name = 'treasure %d' % i
            landmark.respawn_prob = 1.0
            landmark.type = np.random.choice(world.treasure_types)
            landmark.color = world.treasure_colors[landmark.type]
            landmark.alive = True
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.025
            landmark.boundary = False
        world.walls = []
        # make initial conditions
        self.reset_world(world)
        self.reset_cached_rewards()
        return world

    def collectors(self, world):
        return [a for a in world.agents if a.collector]

    def deposits(self, world):
        return [a for a in world.agents if not a.collector]

    def reset_cached_rewards(self):
        self.global_collecting_reward = None
        self.global_holding_reward = None
        self.global_deposit_reward = None

    def post_step(self, world):
        self.reset_cached_rewards()
        for l in world.landmarks:
            if l.alive:
                for a in self.collectors(world):
                    if a.holding is None and self.is_collision(l, a, world):
                        l.alive = False
                        a.holding = l.type
                        a.color = 0.85 * l.color
                        l.state.p_pos = np.array([-999., -999.])
                        break
            else:
                if np.random.uniform() <= l.respawn_prob:
                    bound = 0.95
                    l.state.p_pos = np.random.uniform(low=-bound, high=bound,
                                                      size=world.dim_p)
                    l.type = np.random.choice(world.treasure_types)
                    l.color = world.treasure_colors[l.type]
                    l.alive = True
        for a in self.collectors(world):
            if a.holding is not None:
                for d in self.deposits(world):
                    if d.d_i == a.holding and self.is_collision(a, d, world):
                        a.holding = None
                        a.color = np.array([0.85, 0.85, 0.85])

    def reset_world(self, world):
        # set random initial states
        for i, agent in enumerate(world.agents):
            agent.state.p_pos = np.random.uniform(low=-1, high=1,
                                                  size=world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.holding = None
            if agent.collector:
                agent.color = np.array([0.85, 0.85, 0.85])
        for i, landmark in enumerate(world.landmarks):
            bound = 0.95
            landmark.type = np.random.choice(world.treasure_types)
            landmark.color = world.treasure_colors[landmark.type]
            landmark.state.p_pos = np.random.uniform(low=-bound, high=bound,
                                                     size=world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
            landmark.alive = True
        world.calculate_distances()

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.collector:
            if agent.holding is not None:
                for d in self.deposits(world):
                    if d.d_i == agent.holding and self.is_collision(d, agent, world):
                        return 1
            else:
                for t in self.treasures(world):
                    if self.is_collision(t, agent, world):
                        return 1
        else:  # deposit
            for a in self.collectors(world):
                if a.holding == agent.d_i and self.is_collision(a, agent, world):
                    return 1
        return 0

    def is_collision(self, agent1, agent2, world):
        dist = world.cached_dist_mag[agent1.i, agent2.i]
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def treasures(self, world):
        return world.landmarks

    def reward(self, agent, world):
        main_reward = (self.collector_reward(agent, world) if agent.collector
                       else self.deposit_reward(agent, world))
        return main_reward

    def deposit_reward(self, agent, world):
        rew = 0
        shape = True
        if shape:  # reward can optionally be shaped
            # penalize by distance to closest relevant holding agent
            dists_to_holding = [world.cached_dist_mag[agent.i, a.i] for a in
                                self.collectors(world) if a.holding == agent.d_i]
            if len(dists_to_holding) > 0:
                rew -= 0.1 * min(dists_to_holding)
            else:
                n_visible = 7
                # get positions of all entities in this agent's reference frame
                other_agent_inds = [a.i for a in world.agents if (a is not agent and a.collector)]
                closest_agents = sorted(
                    zip(world.cached_dist_mag[other_agent_inds, agent.i],
                        other_agent_inds))[:n_visible]
                closest_inds = list(i for _, i in closest_agents)
                closest_avg_dist_vect = world.cached_dist_vect[closest_inds, agent.i].mean(axis=0)
                rew -= 0.1 * np.linalg.norm(closest_avg_dist_vect)
        rew += self.global_reward(world)
        return rew

    def collector_reward(self, agent, world):
        rew = 0
        # penalize collisions between collectors
        rew -= 5 * sum(self.is_collision(agent, a, world)
                       for a in self.collectors(world) if a is not agent)
        shape = True
        if agent.holding is None and shape:
            rew -= 0.1 * min(world.cached_dist_mag[t.i, agent.i] for t in
                             self.treasures(world))
        elif shape:
            rew -= 0.1 * min(world.cached_dist_mag[d.i, agent.i] for d in
                             self.deposits(world) if d.d_i == agent.holding)
        # collectors get global reward
        rew += self.global_reward(world)
        return rew

    def global_reward(self, world):
        if self.global_deposit_reward is None:
            self.calc_global_deposit_reward(world)
        if self.global_collecting_reward is None:
            self.calc_global_collecting_reward(world)
        return self.global_deposit_reward + self.global_collecting_reward

    def calc_global_collecting_reward(self, world):
        rew = 0
        for t in self.treasures(world):
            rew += 5 * sum(self.is_collision(a, t, world)
                           for a in self.collectors(world)
                           if a.holding is None)
        self.global_collecting_reward = rew

    def calc_global_deposit_reward(self, world):
        # reward deposits for getting treasure from collectors
        rew = 0
        for d in self.deposits(world):
            rew += 5 * sum(self.is_collision(d, a, world) for a in
                           self.collectors(world) if a.holding == d.d_i)
        self.global_deposit_reward = rew

    def get_agent_encoding(self, agent, world):
        encoding = []
        n_treasure_types = len(world.treasure_types)
        if agent.collector:
            encoding.append(np.zeros(n_treasure_types))
            encoding.append((np.arange(n_treasure_types) == agent.holding))
        else:
            encoding.append((np.arange(n_treasure_types) == agent.d_i))
            encoding.append(np.zeros(n_treasure_types))
        return np.concatenate(encoding)

    def observation(self, agent, world):
        n_visible = 7  # number of other agents and treasures visible to each agent
        # get positions of all entities in this agent's reference frame
        other_agents = [a.i for a in world.agents if a is not agent]
        closest_agents = sorted(
            zip(world.cached_dist_mag[other_agents, agent.i],
                other_agents))[:n_visible]
        treasures = [t.i for t in self.treasures(world)]
        closest_treasures = sorted(
            zip(world.cached_dist_mag[treasures, agent.i],
                treasures))[:n_visible]

        n_treasure_types = len(world.treasure_types)
        obs = [agent.state.p_pos, agent.state.p_vel]
        if agent.collector:
            # collectors need to know their own state bc it changes
            obs.append((np.arange(n_treasure_types) == agent.holding))
        for _, i in closest_agents:
            a = world.entities[i]
            obs.append(world.cached_dist_vect[i, agent.i])
            obs.append(a.state.p_vel)
            obs.append(self.get_agent_encoding(a, world))
        for _, i in closest_treasures:
            t = world.entities[i]
            obs.append(world.cached_dist_vect[i, agent.i])
            obs.append((np.arange(n_treasure_types) == t.type))

        return np.concatenate(obs)
