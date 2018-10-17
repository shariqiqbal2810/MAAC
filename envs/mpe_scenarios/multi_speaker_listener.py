import numpy as np
import seaborn as sns
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 5
        num_listeners = 4
        num_speakers = 4
        num_landmarks = 6
        world.landmark_colors = np.array(
            sns.color_palette(n_colors=num_landmarks))
        world.listeners = []
        for li in range(num_listeners):
            agent = Agent()
            agent.i = li
            agent.name = 'agent %i' % agent.i
            agent.listener = True
            agent.collide = False
            agent.size = 0.075
            agent.silent = True
            agent.accel = 1.5
            agent.initial_mass = 1.0
            agent.max_speed = 1.0
            world.listeners.append(agent)
        world.speakers = []
        for si in range(num_speakers):
            agent = Agent()
            agent.i = si + num_listeners
            agent.name = 'agent %i' % agent.i
            agent.listener = False
            agent.collide = False
            agent.size = 0.075
            agent.movable = False
            agent.accel = 1.5
            agent.initial_mass = 1.0
            agent.max_speed = 1.0
            world.speakers.append(agent)
        world.agents = world.listeners + world.speakers
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.i = i + num_listeners + num_speakers
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.04
            landmark.color = world.landmark_colors[i]
        # make initial conditions
        self.reset_world(world)
        self.reset_cached_rewards()
        return world

    def reset_cached_rewards(self):
        self.pair_rewards = None

    def post_step(self, world):
        self.reset_cached_rewards()

    def reset_world(self, world):
        listen_inds = list(range(len(world.listeners)))
        np.random.shuffle(listen_inds)  # randomize which listener each episode
        for i, speaker in enumerate(world.speakers):
            li = listen_inds[i]
            speaker.listen_ind = li
            speaker.goal_a = world.listeners[li]
            speaker.goal_b = np.random.choice(world.landmarks)
            speaker.color = np.array([0.25,0.25,0.25])
            world.listeners[li].color = speaker.goal_b.color + np.array([0.25, 0.25, 0.25])
            world.listeners[li].speak_ind = i

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        self.reset_cached_rewards()

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        return reward(agent, world)

    def calc_rewards(self, world):
        rews = []
        for speaker in world.speakers:
            dist = np.sum(np.square(speaker.goal_a.state.p_pos -
                                    speaker.goal_b.state.p_pos))
            rew = -dist
            if dist < (speaker.goal_a.size + speaker.goal_b.size) * 1.5:
                rew += 10.
            rews.append(rew)
        return rews

    def reward(self, agent, world):
        if self.pair_rewards is None:
            self.pair_rewards = self.calc_rewards(world)
        share_rews = False
        if share_rews:
            return sum(self.pair_rewards)
        if agent.listener:
            return self.pair_rewards[agent.speak_ind]
        else:
            return self.pair_rewards[agent.goal_a.speak_ind]

    def observation(self, agent, world):
        if agent.listener:
            obs = []
            # give listener index of their speaker
            obs += [agent.speak_ind == np.arange(len(world.speakers))]
            # give listener communication from its speaker
            obs += [world.speakers[agent.speak_ind].state.c]
            # give listener its own position/velocity,
            obs += [agent.state.p_pos, agent.state.p_vel]

            # obs += [world.speakers[agent.speak_ind].state.c]
            # # # give listener index of their speaker
            # # obs += [agent.speak_ind == np.arange(len(world.speakers))]
            # # # give listener all communications
            # # obs += [speaker.state.c for speaker in world.speakers]
            # # give listener its own velocity
            # obs += [agent.state.p_vel]
            # # give listener locations of all agents
            # # obs += [a.state.p_pos for a in world.agents]
            # # give listener locations of all landmarks
            # obs += [l.state.p_pos for l in world.landmarks]
            return np.concatenate(obs)
        else:  # speaker
            obs = []
            # give speaker index of their listener
            obs += [agent.listen_ind == np.arange(len(world.listeners))]
            # speaker gets position of listener and goal
            obs += [agent.goal_a.state.p_pos, agent.goal_b.state.p_pos]

            # # give speaker index of their listener
            # # obs += [agent.listen_ind == np.arange(len(world.listeners))]
            # # # give speaker all communications
            # # obs += [speaker.state.c for speaker in world.speakers]
            # # give speaker their goal color
            # obs += [agent.goal_b.color]
            # # give speaker their listener's position
            # obs += [agent.goal_a.state.p_pos]
            #
            # obs += [speaker.state.c for speaker in world.speakers]
            return np.concatenate(obs)
            
