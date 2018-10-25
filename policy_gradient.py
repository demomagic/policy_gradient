import numpy as np
import cv2
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, Activation

GAMMA = 0.95
GAME_WIDTH, GAME_HEIGHT, STATE_LENGTH = 84, 84, 4
LEARNING_RATE = 0.01

class Agent:
    def __init__(self, actions_size, load_model = False):
        self.actions_size = actions_size
        self.gamma = GAMMA
        
        self.input_state, self.action_distribution, self.model = self.build_network()
        model_network_weights = self.model.trainable_weights
        self.actions_list, self.actions_value, self.loss, self.grads_update = self.build_training(model_network_weights)
        
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        
        if load_model:
            self.model.load('pg_model.h5')
        
        self.ep_obs, self.ep_rs, self.ep_as = [], [], []
        self.episode = 0
        
    def initial_state(self, observation, last_observation):
        new_observation = np.maximum(observation, last_observation)
        gray_observation = cv2.resize(cv2.cvtColor(new_observation, cv2.COLOR_BGR2GRAY),(GAME_WIDTH, GAME_HEIGHT),interpolation = cv2.INTER_CUBIC)
        state = [np.uint8(gray_observation) for _ in range(STATE_LENGTH)]
        return np.stack(state, axis=2)
    
    def build_network(self):
        model = Sequential()
        model.add(Conv2D(16, kernel_size = (8, 8), strides = (4, 4), padding = 'valid', activation = 'relu', input_shape=(GAME_WIDTH, GAME_HEIGHT, STATE_LENGTH)))
        model.add(Conv2D(32, kernel_size = (8, 8), strides = (4, 4), padding = 'valid', activation = 'relu'))
        model.add(Flatten())
        model.add(Dense(256, activation = 'relu'))
        model.add(Dense(self.actions_size, activation = 'sigmoid'))
        model.add(Activation(activation = 'softmax'))
        
        input_state = tf.placeholder(tf.float32, [None, GAME_WIDTH, GAME_HEIGHT, STATE_LENGTH])
        action_distribution = model(input_state)
        
        return input_state, action_distribution, model
    
    def build_training(self, model_network_weights):
        actions_list = tf.placeholder(tf.int32, [None, ])
        actions_value = tf.placeholder(tf.float32, [None, ])
        
        neg_log_prob = tf.reduce_sum(-tf.log(self.action_distribution)*tf.one_hot(actions_list, self.actions_size), axis=1)
        loss = tf.reduce_mean(neg_log_prob * actions_value)
        
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        grads_update = optimizer.minimize(loss, var_list = model_network_weights)
        
        return actions_list, actions_value, loss, grads_update
    
    def train_network(self):
        self.episode += 1
        episode_norm_reward = self.discount_and_norm_rewards()
        loss, _ = self.sess.run([self.loss, self.grads_update], feed_dict={
            self.input_state: np.float32(self.ep_obs),
            self.actions_list: np.array(self.ep_as),
            self.actions_value: episode_norm_reward
        })

        if self.episode % 100 == 0:
            self.model.save('pg_model.h5')

        print('Episode: %d - Score: %f. - Loss: %f.' % (self.episode, np.sum(self.ep_rs), loss))
        self.ep_obs, self.ep_rs, self.ep_as = [], [], []
        
    def store_transition(self, state, action, reward):
        self.ep_obs.append(state)
        self.ep_as.append(action)
        self.ep_rs.append(reward)
        
    def get_action(self, state):
        act_prob = self.action_distribution.eval(feed_dict = {self.input_state: [np.float32(state)]}).flatten()
        prob = act_prob / np.sum(act_prob)
        action = np.random.choice(self.actions_size, 1, p = prob)[0]
        return action
    
    def discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs
    
def process_observation(observation, last_observation):
    new_observation = np.maximum(observation, last_observation)
    gray_observation = cv2.resize(cv2.cvtColor(new_observation, cv2.COLOR_BGR2GRAY), (GAME_WIDTH, GAME_HEIGHT), interpolation = cv2.INTER_CUBIC)
    return np.reshape(np.uint8(gray_observation), (GAME_WIDTH, GAME_HEIGHT, 1))
