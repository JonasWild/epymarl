import numpy as np

from src.components.episode_buffer import ReplayBuffer


class MetaReplayBuffer(ReplayBuffer):
    def __init__(self, scheme, groups, buffer_size, max_seq_length, preprocess=None, device="cpu"):
        super(MetaReplayBuffer, self).__init__(scheme, groups, buffer_size, max_seq_length, preprocess=preprocess,
                                               device=device)
        self.rewards = np.zeros(buffer_size)

    def insert_episode_batch(self, ep_batch):
        custom_rewards = ep_batch["reward"][:, :-1]
        if custom_rewards is not None and custom_rewards.sum(dim=1).item() == 0:
            if np.random.rand() > 0.1:  # 90% chance to ignore zero-reward ep_batch
                return

        if self.buffer_index + ep_batch.batch_size <= self.buffer_size:
            if custom_rewards is not None:
                self.rewards[self.buffer_index:self.buffer_index + ep_batch.batch_size] = custom_rewards.sum(
                    dim=1).cpu().numpy()

            self.update(ep_batch.data.transition_data,
                        slice(self.buffer_index, self.buffer_index + ep_batch.batch_size),
                        slice(0, ep_batch.max_seq_length),
                        mark_filled=False)
            self.update(ep_batch.data.episode_data,
                        slice(self.buffer_index, self.buffer_index + ep_batch.batch_size))
            self.buffer_index = (self.buffer_index + ep_batch.batch_size)
            self.episodes_in_buffer = max(self.episodes_in_buffer, self.buffer_index)
            self.buffer_index = self.buffer_index % self.buffer_size
            assert self.buffer_index < self.buffer_size
        else:
            # Check if the ep_batch to be removed has non-zero reward
            batch_to_remove_reward = self.rewards[self.buffer_index]

            if batch_to_remove_reward != 0 and np.random.rand() > 0.1:
                # 90% chance to keep the non-zero reward batch and remove the next zero-reward batch instead
                zero_reward_indices = np.where(self.rewards == 0)[0]
                if len(zero_reward_indices) > 0:
                    batch_to_remove_index = zero_reward_indices[0]  # Get the index of the first zero-reward batch
                    # Logic to remove the zero-reward batch
                    # This may need to be adapted depending on how your buffer and data are structured
                    self.rewards = np.delete(self.rewards, batch_to_remove_index)
                    self.rewards = np.append(self.rewards, 0)  # Append zero to keep the same size

            buffer_left = self.buffer_size - self.buffer_index
            super(MetaReplayBuffer, self).insert_episode_batch(ep_batch[0:buffer_left, :])
            super(MetaReplayBuffer, self).insert_episode_batch(ep_batch[buffer_left:, :])
