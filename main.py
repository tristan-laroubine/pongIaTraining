import gym
import imageio as imageio
import numpy as np
import pickle
from gym.utils.play import play
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils

def main():
    env = gym.make('Pong-v4')
    env.reset()
    play(env, zoom=3, fps=12, callback=mycallback)
    env.close()

XarrayFrame = []
YarrayFrame = []
RarrayFrame = []


def vectoriseGameOutput(obs_t):
    return obs_t[34:194:4, 12:148:2, 1]


def testIA():
    env = gym.make('Pong-v4')
    env.reset()
    model = get_agent()
    obs_t, obs_tp1, action, rew, done, info = env.step(env.action_space.sample())  # take a random action
    while done is not True:
        x_data = vectoriseGameOutput(obs_t)
        obs_t, obs_tp1, action, rew, done, info = env.step(model.predict(x_data))
        if (rew != 0):
            print(rew)

    env.close()

def get_agent():
    x_train = np.asarray(pickle.load(open("X.p", "rb")))
    y_train = np.asarray(pickle.load(open("Y.p", "rb")))

    model = keras.Sequential()
    model.add(layers.Dense(16, input_dim=1, activation='relu'))
    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.Dense(3))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.fit(x_train, y_train, epochs=100, validation_split=0.33)
    #ValueError: Input 0 of layer sequential is incompatible with the layer: expected axis -1 of input shape to have value 1 but received input with shape (None, 2720)

    return model



def mycallback(obs_t, obs_tp1, action, rew, done, info):
    print("action = ", action, " reward = ", rew, "done = ", done)
    # imageio.imwrite("jeu.jpg", obs_t[34:194:4, 12:148:2, 1])
    matrix_fram = obs_t[34:194:4, 12:148:2, 1]

    XarrayFrame.append(np.array(matrix_fram).flatten())
    YarrayFrame.append(action)
    RarrayFrame.append(rew)
    if (done):
        pickle.dump(XarrayFrame, open("X.p", "wb"))
        pickle.dump(YarrayFrame, open("Y.p", "wb"))
        pickle.dump(RarrayFrame, open("R.p", "wb"))
        exit(0)





if __name__ == '__main__':
    testIA()
