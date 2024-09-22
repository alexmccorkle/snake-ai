import torch
import torch.nn as nn # Neural Network
import torch.optim as optim
import torch.nn.functional as F 
import numpy as np
import os

class Linear_QNet(nn.Module): # Linear Neural Network for Q-Learning
  def __init__(self, input_size, hidden_size, output_size): # Constructor:
    super().__init__() # Inherit from the nn.Module class
    self.linear1 = nn.Linear(input_size, hidden_size) # The first linear layer
    self.linear2 = nn.Linear(hidden_size, output_size) # The second linear layer

  def forward(self, x): # Forward function which is called when we pass data through the network
    x = F.relu(self.linear1(x)) # Pass the data through the first layer and apply the ReLU activation function
    x = self.linear2(x) # Pass the data through the second layer

    return x # Return the output

  # def save(self, file_name='model.pth'):
  #   model_folder_path = './model'
  #   if not os.path.exists(model_folder_path): # If the model folder doesn't exist
  #     os.makedirs(model_folder_path)

  #   file_name = os.path.join(model_folder_path, file_name)
  #   torch.save(self.state_dict(), file_name) # Save the model's state dictionary to the file

  def save_checkpoint(self, optimizer, n_games, record, total_score, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": self.state_dict(),
        "optimizer": optimizer.state_dict(),
        "games_played": n_games,
        # Other data:
        "record": record,
        "total_score": total_score
    }
    torch.save(checkpoint, filename)
    print("=> Checkpoint saved")

  def load_checkpoint(self, optimizer, filename="my_checkpoint.pth.tar"):
    if os.path.isfile(filename) :
      checkpoint = torch.load(filename, weights_only=True)
      self.load_state_dict(checkpoint["state_dict"])  
      optimizer.load_state_dict(checkpoint["optimizer"])
      games_played = checkpoint.get("games_played", 0)
      record = checkpoint.get("record", 0)
      total_score = checkpoint.get("total_score", 0)
      print("=> Checkpoint loaded")
      return games_played, record, total_score
    else:
      print("=> No checkpoint found")
      return 0, 0, 0
class QTrainer: # Q-Learning Trainer
  def __init__(self, model, lr, gamma): # Constructor
    self.model = model # The model we are training
    self.lr = lr
    self.gamma = gamma

    self.optimizer = optim.Adam(model.parameters(), lr=self.lr) # Adam optimizer
    self.criterion = nn.MSELoss() # Mean Squared Error Loss

  def train_step(self, state, action, reward, next_state, done):
    state = torch.tensor(np.array(state), dtype=torch.float)
    next_state = torch.tensor(np.array(next_state), dtype=torch.float)
    action = torch.tensor(np.array(action), dtype=torch.long) # Long because it's an index
    reward = torch.tensor(np.array(reward), dtype=torch.float)

    if len(state.shape) == 1: # If the state is a 1D tensor
      # (1, x) 
      state = torch.unsqueeze(state, 0) # Add a dimension at the beginning
        # We do this because the model expects a 2D tensor
      next_state = torch.unsqueeze(next_state, 0)
      action = torch.unsqueeze(action, 0)
      reward = torch.unsqueeze(reward, 0)
      done = (done, ) # Tuple

      # 1: Predicted Q-Values
        # Q-Values are the expected rewards for each action
      pred = self.model(state)

      target = pred.clone() # Clone the predicted tensor

      for idx in range(len(done)):
        Q_new = reward[idx]
        if not done[idx]:
          Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx])) # Q-Learning formula

        target[idx][torch.argmax(action[idx]).item()] = Q_new # Update the target Q-Value



      # 2: Target Q-Values: Q_new = (reward + gamma * max(next_predicted_q_value)) -> only do this if not done
      # pred.clone() so that we don't change the original tensor
      # pred[argsmax(action)] = Q_new
      self.optimizer.zero_grad() # Zero the gradients, so that they don't accumulate
      loss = self.criterion(target, pred)
      loss.backward() # Backpropagation, compute the gradients
      
      self.optimizer.step() # Update the weights 

  def save_checkpoint(self, filename="trainer_checkpoint.pth.tar"):
    print("=> Saving trainer checkpoint")
    checkpoint = {
        "optimizer": self.optimizer.state_dict()
    }
    torch.save(checkpoint, filename)
    print("=> Trainer checkpoint saved")

  def load_checkpoint(self, filename="trainer_checkpoint.pth.tar"):
      if os.path.isfile(filename):
          print("=> Loading trainer checkpoint")
          checkpoint = torch.load(filename, weights_only=True)
          self.optimizer.load_state_dict(checkpoint["optimizer"])
          print("=> Trainer checkpoint loaded")
      else:
          print("=> No trainer checkpoint found")


