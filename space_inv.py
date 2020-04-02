import random
from collections import deque
import argparse
import time
from time import sleep
from subprocess import Popen

import cv2
import numpy as np
import matplotlib.pyplot as plt

import gym
from gym.utils.play import play
from gym.wrappers.monitoring.video_recorder import VideoRecorder

import tensorflow as tf
import keras as keras
from keras.layers import Dense, Conv2D, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import Adam

class DQN:
    def __init__(self, state_size, action_size, model_depth=4):
        self.state_size = state_size
        self.action_size = action_size
        self.model_depth = model_depth
        self.memory = deque(maxlen=4000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.model = self._create_model()

    def _create_model(self):
        model = Sequential()
        model.add(Conv2D(16, (3,3), input_shape=(self.state_size[0], self.state_size[1], self.model_depth), activation='relu'))
        model.add(Dropout(0.25))

        model.add(Conv2D(32, (3,3), activation='relu'))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def add_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        pred = self.model.predict(np.expand_dims(state, axis=0))
        return np.argmax(pred[0])

    def replay(self):
        batch = random.sample(self.memory, self.batch_size)
        states, targets_f = [], []
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(np.expand_dims(next_state, axis=0)))

            target_f = self.model.predict(np.expand_dims(state, axis=0))
            target_f[0][action] = target
            states.append(state)
            targets_f.append(target_f[0])

        history = self.model.fit(np.array(states), np.array(targets_f), epochs=1, verbose = 0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return history

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

def preprocess(obv):
    obv = cv2.cvtColor(cv2.resize(obv, (84, 110)), cv2.COLOR_BGR2GRAY)
    obv = obv[26:, :]
    ret, obv = cv2.threshold(obv, 1, 255, cv2.THRESH_BINARY)
    return np.reshape(obv, (84, 84, 1))

def short_to_state(short):
    return np.reshape(np.array(short), (84, 84, 4))


class Space_Invaders:

    def __init__(self):
        self.short_memory = 4
        self.max_games = 10000
        self.no_of_steps = 10000
        self.chatbot_env = gym.make('SpaceInvaders-v0')
        self.state_size = (84,84)
        self.action_size = self.chatbot_env.action_space.n
        self.agent = DQN(self.state_size, self.action_size, self.short_memory)
        self.agent.load("./space_invaders_trained.h5")
        
    def train(self, resume_weights):
        if resume_weights != None:
            self.agent.load(resume_weights)
        for game in range(self.max_games):
            obv = preprocess(chatbot_env.reset())
            short_mem = deque([obv]*self.short_memory, maxlen=self.short_memory)
            state = short_to_state(short_mem)
            for t in range( self.max_games):
                action = agent.act(state)
                obv, reward, done, info = self.chatbot_env.step(action)
                obv = preprocess(obv)
                reward = reward if not done else -20
                short_mem.append(obv)
                next_state = short_to_state(short_mem)
                self.agent.add_memory(state, action, reward, next_state, done)
                state = next_state
                if done:
                    print("reward = " + reward)
                    break
                if len(agent.memory) > agent.batch_size:
                    history = agent.replay()
            if game % 50 == 0:
                self.agent.save("./space_invaders_"+str(game)+".h5")

    def duel(self):
        Popen(['python3', 'keyboard_agent.py', 'SpaceInvaders-v0'])
        self.chatbot_play()

    def chatbot_play(self):
        obv = preprocess(self.chatbot_env.reset())
        short_mem = deque([obv]*self.short_memory, maxlen=self.short_memory)
        state = short_to_state(short_mem)
        done = False
        while not done:
            sleep(0.04615)
            action = self.agent.act(state)
            obv, reward, done, info = self.chatbot_env.step(action)
            rgb = self.chatbot_env.render()
            obv = preprocess(obv)
            short_mem.append(obv)
            state = short_to_state(short_mem)
        self.chatbot_env.close()


if __name__ == '__main__':
    game = Space_Invaders()
    game.duel()