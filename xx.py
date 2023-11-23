import numpy as np

from custom.myutils.myutils import pad_second_dim

d = 15
actions_episode = [np.zeros(shape=(32, 2, 15)).tolist(), np.zeros(shape=(16, 2, 15)).tolist()]

print(np.array(actions_episode).shape)

padded_actions = pad_second_dim(actions_episode, [[0.0] * 15, [0.0] * 15])

print(np.array(padded_actions).shape)
