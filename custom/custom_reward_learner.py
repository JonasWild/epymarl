import torch
import torch.nn as nn
import torch.optim as optim

from src.components.episode_buffer import EpisodeBatch


class CustomRewardLearner(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, is_target_model=False, use_target_network=False):
        super(CustomRewardLearner, self).__init__()

        # Define the neural network layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # Set up the loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

        self.use_target_network = use_target_network

        # Initialize the target network and set its weights equal to the policy network
        if self.use_target_network and not is_target_model:
            self.target_model = self.__class__(input_dim, hidden_dim, output_dim, is_target_model=True)

            model_state = {k: v for k, v in self.state_dict().items() if 'target_model' not in k}
            self.target_model.load_state_dict(model_state)

            self.target_model.eval()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x

    def train_model(self, batch: EpisodeBatch):
        # Convert transitions to PyTorch tensors
        states = batch["state"][:, :-1]
        custom_rewards = batch["reward"][:, :-1]

        # Forward pass
        predicted_rewards = self(states)

        loss = 0
        if self.use_target_network:
            # Use the target network to calculate the target value for loss calculation
            with torch.no_grad():  # no need to calculate gradients for the target network
                target_rewards = self.target_model(states)

            loss = self.criterion(predicted_rewards.squeeze(), custom_rewards.squeeze() + target_rewards.squeeze())
        else:
            loss = self.criterion(predicted_rewards.squeeze(), custom_rewards.squeeze())

        # Zero gradients, backward pass, optimizer step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self, tau=0.001):
        """
        Soft update the target network weights with the policy network weights.

        :param tau: the soft update coefficient, a value between 0 and 1.
        """
        for target_param, policy_param in zip(self.target_model.parameters(), self.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

    def predict_reward(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state)
            predicted_reward = self(state)
        return predicted_reward.numpy()
