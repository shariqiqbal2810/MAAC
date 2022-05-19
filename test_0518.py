'''
# from multiagent.environment import MultiAgentEnv
from make_env import make_env
import PIL
# from pyvirtualdisplay import Display

# display = Display(visible=True, size=(400, 300))
# display.start()
env = make_env('simple_tag')

for i in range(10):
    env.reset()
    env.render()
    # PIL.Image.fromarray(env.render(mode='rgb_array'))

# env.render()
# print(env)

env.close()
'''
import gym

env = gym.make('CartPole-v0')

for i in range(20):
    env.render()

	# observation = env.reset()
	# for j in range(100):
	# 	# image = env.render(mode="rgb_array")
	# 	action = env.action_space.sample()

env.close()



