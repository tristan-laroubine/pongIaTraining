import gym
import imageio as imageio
import numpy as np
import pickle
from gym.utils.play import play


def main():
    env = gym.make('Pong-v4')
    env.reset()
    play(env, zoom=3, fps=12, callback=mycallback)
    env.close()

XarrayFrame = []
YarrayFrame = []
RarrayFrame = []

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
    main()
