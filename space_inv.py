import random
import cv2
import numpy as np
from collections import deque
from time import sleep
from subprocess import Popen
import gym
import tensorflow as tf
import keras as keras
from keras.layers import Dense, Conv2D, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import Adam

# Deep Q Learning Network made following this tutorial
# https://keon.github.io/deep-q-learning/
class DQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
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
        # Input shape 84x84x4
        model.add(Conv2D(16, (3,3), input_shape=(self.state_size[0], self.state_size[1], 4), activation='relu'))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, (3,3), activation='relu'))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def memorise(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        pred = self.model.predict(np.expand_dims(state, axis=0))
        return np.argmax(pred[0])

    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        states, targets_f = [], []
        for state, action, reward, next_state, done in minibatch:
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

    # Added wrapper for save_weights and load_weights functions
    def save_weights(self, filename):
        self.model.save_weights(filename)

    def load_weights(self, filename):
        self.model.load_weights(filename)

# Adapted the image pre-processing function from https://github.com/floodsung/DQN-Atari-Tensorflow/blob/master/AtariDQN.py
def preprocess(frame):
    frame = cv2.cvtColor(cv2.resize(frame, (84, 110)), cv2.COLOR_BGR2GRAY)
    frame = frame[26:, :]
    ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
    return np.reshape(frame, (84, 84, 1))

# Reshaping the array to 84x84x4 (input shape)
def reshape(arr):
    return np.reshape(np.array(arr), (84, 84, 4))

# Formatted the whole game as an object which starts 2 windows (one for user and 1 for chatbot agent)
# To start the game do obj.duel()
# To train the agent do obj.train(resume_weights_filepath)
class Space_Invaders:
    # Initialising training parameters and a gym environment 
    def __init__(self):
        self.model_depth = 4
        self.max_games = 10000
        self.no_of_steps = 10000
        self.chatbot_env = gym.make('SpaceInvaders-v0')
        self.state_size = (84,84)
        self.action_size = self.chatbot_env.action_space.n
        self.agent = DQN(self.state_size, self.action_size)
        # Loading the saved weights
        self.agent.load_weights("./space_invaders_trained.h5")
    
    # Agent training function, takes weights filepath as input
    def train(self, resume_weights):
        # If the filepath is specified, load the weights and continue training
        # Otherwise start training from scratch
        if resume_weights != None:
            self.agent.load(resume_weights)
        # Agent training block
        for game in range(self.max_games):
            # Resetting the environment to start a new game
            next_frame = preprocess(chatbot_env.reset())
            # Processing 4 frames at once
            mem_frames = deque([next_frame]*self.model_depth, maxlen=self.model_depth)
            state = reshape(mem_frames)
            for i in range(self.max_games):
                # Feed the agent with action
                action = agent.act(state)
                next_frame, reward, done, info = self.chatbot_env.step(action)
                # Preprocess the result frame for next iteration
                next_frame = preprocess(next_frame)
                # Check if the game is beaten, and set the reward
                # print('Score (reward) = ' + str(reward))
                if not done:
                    reward = -20
                # Append the output frames to memory and generate the next statte
                mem_frames.append(next_frame)
                next_state = reshape(mem_frames)
                # Memorise the data and set up the state for next iteration
                self.agent.memorise(state, action, reward, next_state, done)
                state = next_state
                # If the game is completed, print the reward
                if done:
                    print("Reward = " + reward)
                    break
                # Train the model based on previous memorised experiences
                if len(agent.memory) > agent.batch_size:
                    history = agent.replay()

            # Saving every 50th episode        
            if game % 50 == 0:
                self.agent.save_weights("./space_invaders_"+str(game)+".h5")

    # Function created to start a duel between user and chatbot
    def duel(self):
        # Starting the OpenAI keyboard_agent to let the user play 
        Popen(['python3', 'keyboard_agent.py', 'SpaceInvaders-v0'])
        # User game takes some time to load while the chatbot one starts immediately
        # Therefore, wait for a bit before starting the chatbot game
        sleep(2000)
        self.chatbot_play()

    # Function created to utilise the trained agent to play the game
    def chatbot_play(self):
        # Preprocessing and preparation is done the same way as in the training
        next_frame = preprocess(self.chatbot_env.reset())
        mem_frames = deque([next_frame]*self.model_depth, maxlen=self.model_depth)
        state = reshape(mem_frames)
        done = False
        # Keep playing until the game is not done
        while not done:
            # Sleeping for 0.04615 makes the game 24fps
            sleep(0.04615)
            action = self.agent.act(state)
            next_frame, reward, done, info = self.chatbot_env.step(action)
            rgb = self.chatbot_env.render()
            # The default sizes of env.render() windows are really small
            # The renders could be saved in a variable and then scaled up and 
            # shown with something like matplotlib or SimpleImageViewer
            next_frame = preprocess(next_frame)
            mem_frames.append(next_frame)
            state = reshape(mem_frames)
        # Close the environment when the game is done
        self.chatbot_env.close()

# Create a new object of space invaders 
# And start a duel (human vs chatbot)
if __name__ == '__main__':
    game = Space_Invaders()
    game.duel()