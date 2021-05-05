# Project created by Lucie MORANT & Tristan LAROUBINE

import gym
import imageio as imageio
import numpy as np
import pickle
import pygame
import keras
from gym.utils.play import play
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils

# variable used to store the data of a hand played game
XarrayFrame = []
YarrayFrame = []
RarrayFrame = []


def records_game_played_by_hand():  # saves the history of a game
    env = gym.make('Pong-v4')
    env.reset()
    play(env, zoom=3, fps=12, callback=saveFrameGame)
    env.close()


def saveFrameGame(obs_t, obs_tp1, action, rew, done, info):
    print("action = ", action, " reward = ", rew, "done = ", done)
    # imageio.imwrite("jeu.jpg", obs_t[34:194:4, 12:148:2, 1])
    matrix_frame = vectorization_game_output(obs_t, 1)
    matrix_frame_before = vectorization_game_output(obs_tp1, 1)
    if action != 0:
        action = action - 1
    XarrayFrame.append(np.array(matrix_frame - matrix_frame_before).flatten())
    YarrayFrame.append(action)
    RarrayFrame.append(rew)
    if done:
        pickle.dump(XarrayFrame, open("X.p", "wb"))
        pickle.dump(YarrayFrame, open("Y.p", "wb"))
        pickle.dump(RarrayFrame, open("R.p", "wb"))
        exit(0)


def vectorization_game_output(obs_t, channel):
    return obs_t[34:194:4, 12:148:2, channel]


def test_IA():
    env = gym.make('Pong-v4')
    env.reset()
    model = get_agent()
    obs_t, rew, done, inf = env.step(env.action_space.sample())  # take a random action
    obs_t1 = obs_t
    while done is not True:
        x_data = vectorization_game_output(obs_t, 1) - vectorization_game_output(obs_t1, 1)
        x_data = np.array(x_data).flatten()
        x_data = x_data.reshape(1, (40 * 68))
        action = np.argmax(model.predict(x_data), axis=1)
        if action != 0:
            action += 1
        obs_t1 = obs_t
        obs_t, rew, done, inf = env.step(action)
        env.render()
        if rew != 0:
            print(rew)

    env.close()


def get_agent(name_of_model="model-3_200-50-3"):
    if False:  # is exist
        return keras.models.load_model('./saved_model.pb')
    else:
        x_train = np.asarray(pickle.load(open("X.p", "rb")))
        print(x_train.shape)
        y_train = np.asarray(pickle.load(open("Y.p", "rb")))
        nb_class = 3
        y_train = keras.utils.to_categorical(y_train, nb_class)
        model = keras.Sequential()
        model.add(layers.Dense(200, input_dim=68 * 40, activation='sigmoid'))
        model.add(layers.Dense(50, activation='sigmoid'))
        model.add(layers.Dense(3, activation='softmax'))
        model.summary()
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        from tensorflow.python.keras.callbacks import EarlyStopping
        # need to understand the fuck
        ourCallback = EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=20, verbose=0, mode='auto',
                                    baseline=None, restore_best_weights=False)
        model.fit(x_train, y_train, epochs=100, batch_size=128, validation_split=0.2, callbacks=[ourCallback])

        # model.fit(x_train, y_train, epochs=100, validation_split=0.33)
        #model.fit(x_train, y_train, epochs=100)
        model.save(".")
        return model


if __name__ == '__main__':
    needCreateRecord = False
    if needCreateRecord:
        print("start : records_game_played_by_hand")
        records_game_played_by_hand()
    else:
        test_IA()
