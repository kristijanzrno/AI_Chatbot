from __future__ import print_function
import sys, gym, time

# This script is made by OpenAI 
# https://github.com/openai/gym/blob/master/examples/agents/keyboard_agent.py

# Modified by Kristijan Zrno, March, 2020 (modifications are commented)

#
# Test yourself as a learning agent! Pass environment name as a command-line argument, for example:
#
# python keyboard_agent.py SpaceInvadersNoFrameskip-v4
#

env = gym.make('LunarLander-v2' if len(sys.argv)<2 else sys.argv[1])

if not hasattr(env.action_space, 'n'):
    raise Exception('Keyboard agent only supports discrete action spaces')
ACTIONS = env.action_space.n
SKIP_CONTROL = 0    # Use previous control decision SKIP_CONTROL times, that's how you
                    # can test what skip is still usable.

human_agent_action = 0
human_wants_restart = False
human_sets_pause = False

def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause
    if key==0xff0d: human_wants_restart = True
    if key==32: human_sets_pause = not human_sets_pause
    a = int( key - ord('0') )
    # Modified by Kristijan Zrno
    # Remapping default keys to A D S (according to ASCII table)
    if key == 97:
        a = 3
    elif key == 100:
        a = 2
    elif key == 115:
        a = 1
    if a <= 0 or a >= ACTIONS: return
    human_agent_action = a

def key_release(key, mod):
    global human_agent_action
    a = int( key - ord('0') )
    # Modified by Kristijan Zrno
    # Remapping default keys to A D S (according to ASCII table)
    if key == 97:
        a = 3
    elif key == 100:
        a = 2
    elif key == 115:
        a = 1
    if a <= 0 or a >= ACTIONS: return
    if human_agent_action == a:
        human_agent_action = 0

env.render()
env.unwrapped.viewer.window.on_key_press = key_press
env.unwrapped.viewer.window.on_key_release = key_release

def rollout(env):
    global human_agent_action, human_wants_restart, human_sets_pause
    human_wants_restart = False
    obser = env.reset()
    skip = 0
    total_reward = 0
    total_timesteps = 0
    while 1:
        if not skip:
            a = human_agent_action
            total_timesteps += 1
            skip = SKIP_CONTROL
        else:
            skip -= 1

        obser, r, done, info = env.step(a)
        if r != 0:
            print("reward %0.3f" % r)
        total_reward += r
        window_still_open = env.render()
        if window_still_open==False: return False
        if done: break
        if human_wants_restart: break
        # Modified by Kristijan Zrno
        # Using the time.sleep(0.04615) to simulate 24fps
        while human_sets_pause:
            env.render()
            time.sleep(0.04615)
        time.sleep(0.04615)
    # Modified by Kristijan Zrno
    # Modified environment to close when the game is finished        
    env.close()
    exit()

played = False
while not played:
    window_still_open = rollout(env)
    played = True
    if window_still_open==False: break

