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
        
        # Device selection priority: CUDA > MPS > CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)} for training.")
            # CUDA optimization settings
            if torch.cuda.get_device_capability()[0] >= 7:  # For newer GPUs
                torch.backends.cudnn.benchmark = True
                print("CUDA optimizations enabled.")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS (Apple Silicon GPU) for training.")
        else:
            self.device = torch.device("cpu")
            print("GPU not available, using CPU for training.")

        self.model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Display device information
        if self.device.type == 'cuda':
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"GPU Memory Available: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / 1e9:.1f} GB")
        
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

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
        Trains the model using experience replay with improved batch processing.
        """
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        
        # Batch processing for better GPU utilization
        states = torch.FloatTensor([experience[0] for experience in minibatch]).to(self.device)
        actions = torch.LongTensor([experience[1] for experience in minibatch]).to(self.device)
        rewards = torch.FloatTensor([experience[2] for experience in minibatch]).to(self.device)
        next_states = torch.FloatTensor([experience[3] for experience in minibatch]).to(self.device)
        dones = torch.BoolTensor([experience[4] for experience in minibatch]).to(self.device)

        # Get current Q-values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Get next Q-values
        with torch.no_grad():
            next_q_values = self.model(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * (~dones))

        # Calculate loss
        loss = torch.nn.functional.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability (especially useful with CUDA)
        if self.device.type == 'cuda':
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        """
        Loads a pre-trained model with device compatibility.
        """
        try:
            # Load with automatic device mapping
            state_dict = torch.load(name, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Model loaded successfully from {name} on {self.device}")
        except Exception as e:
            print(f"Error loading model from {name}: {e}")
            # Fallback to CPU loading
            state_dict = torch.load(name, map_location='cpu')
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            print(f"Model loaded on CPU and moved to {self.device}")

    def save(self, name):
        """
        Saves the current model with device information.
        """
        # Move to CPU before saving for better compatibility
        model_state = self.model.state_dict()
        if self.device.type != 'cpu':
            model_state = {k: v.cpu() for k, v in model_state.items()}
        
        torch.save(model_state, name)
        print(f"Model saved to {name} (device: {self.device})")
