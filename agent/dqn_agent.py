
import random
import numpy as np
import torch
import torch.optim as optim
from collections import deque
from agent.dqn_model import DQN

class DQNAgent:
    """
    DQN Agent for playing Quarto.
    """
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # Check for MPS availability
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS (GPU) for training.")
        else:
            self.device = torch.device("cpu")
            print("MPS not available, using CPU for training.")

        self.model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def remember(self, state, action, reward, next_state, done):
        """
        Stores experience in the replay memory.
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, valid_actions=None):
        """
        Selects an action using an epsilon-greedy policy.
        If valid_actions is provided, only selects from those actions.
        """
        if valid_actions is None:
            valid_actions = list(range(self.action_size))

        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.model.eval() # Set model to evaluation mode
        with torch.no_grad():
            act_values = self.model(state)
        self.model.train() # Set model back to training mode

        # Mask invalid actions by setting their Q-values to a very low number
        masked_act_values = act_values.cpu().data.numpy().flatten()
        for i in range(self.action_size):
            if i not in valid_actions:
                masked_act_values[i] = -np.inf # Set to negative infinity
        
        return np.argmax(masked_act_values)

    def replay(self, batch_size):
        """
        Trains the model using experience replay.
        """
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            reward = torch.FloatTensor([reward]).to(self.device)
            action = torch.LongTensor([action]).to(self.device)

            if done:
                target = reward
            else:
                # Get the Q-values for the next_state from the model
                next_q_values = self.model(next_state)
                # Select the maximum Q-value for the next state
                max_next_q_value = torch.max(next_q_values)
                target = reward + self.gamma * max_next_q_value

            # Get the predicted Q-value for the action taken
            prediction = self.model(state).gather(0, action)

            # Calculate loss
            loss = (prediction - target).pow(2).mean()
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        """
        Loads a pre-trained model.
        """
        self.model.load_state_dict(torch.load(name, map_location=self.device))

    def save(self, name):
        """
        Saves the current model.
        """
        torch.save(self.model.state_dict(), name)
