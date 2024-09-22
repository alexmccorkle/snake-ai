import os
import torch
import random
import numpy as np
from collections import deque # A deque is a list optimized for adding and removing items

from game import SnakeGameAI, Direction, Point # The necessary imports for the agent to work
from model import Linear_QNet, QTrainer # The model and trainer classes
from helper import plot # The plot function to plot the scores

MAX_MEMORY = 100_000 # The maximum number of items we store
BATCH_SIZE = 1000 # The number of items we sample from memory to learn

LR = 0.001 # Learning rate
# Changing this value will change how much the agent learns from each iteration
# A higher value will make the agent learn faster, but it might become unstable

class Agent: 

  def __init__(self):
    self.n_games = 0
    self.epsilon = 0 # Epsilon is the probability of choosing a random action
    self.gamma = 0.9 # Gamma is the discount factor
    self.memory = deque(maxlen=MAX_MEMORY) 
    # If we exceed memory, it will remove the oldest item (popleft())
    self.model = Linear_QNet(11, 256, 3) # 11 input nodes, 256 hidden nodes, 3 output nodes
      # Hidden nodes ELI5: The number of nodes in the middle layer of the neural network, the more nodes the more complex patterns it can learn
    self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
    self.optimizer = self.trainer.optimizer


  def get_state(self, game): 
    # Return the current state of the game
    head = game.snake[0] # The head of the snake
    # Creates 4 points around the head so that we can check for danger and food in it's surroundings
    point_l = Point(head.x - 20, head.y)
    point_r = Point(head.x + 20, head.y)
    point_u = Point(head.x, head.y - 20)
    point_d = Point(head.x, head.y + 20)

    # Direction the snake is going:
    dir_l = game.direction == Direction.LEFT
    dir_r = game.direction == Direction.RIGHT
    dir_u = game.direction == Direction.UP
    dir_d = game.direction == Direction.DOWN

    state = [
      # Danger Straight:
      (dir_r and game.is_collision(point_r)) or # If the snake is going right, and there is a collision to the right = danger straight ahead
      (dir_l and game.is_collision(point_l)) or # And so on...
      (dir_u and game.is_collision(point_u)) or
      (dir_d and game.is_collision(point_d)),

      # Danger Right:
      (dir_u and game.is_collision(point_r)) or # If the snake is going up and the point to the right is a collision = danger right
      (dir_d and game.is_collision(point_l)) or # and so on...
      (dir_l and game.is_collision(point_u)) or
      (dir_r and game.is_collision(point_d)),

      # Danger Left:
      (dir_d and game.is_collision(point_r)) or
      (dir_u and game.is_collision(point_l)) or
      (dir_r and game.is_collision(point_u)) or
      (dir_l and game.is_collision(point_d)),
      
      # Move direction: Only one of these will be true
      dir_l,
      dir_r,
      dir_u,
      dir_d,

      # Food location: If the food is to the right, below, or above the snake
      game.food.x < game.head.x, # Food is to the left
      game.food.x > game.head.x, # Food is to the right
      game.food.y < game.head.y, # Food is above
      game.food.y > game.head.y # Food is below
      ]
    
    return np.array(state, dtype=int) # Return the state as a numpy array, and set the data type to int so that bools become 0 or 1


  def remember(self, state, action, reward, next_state, done):
    # Store the state, action, reward, next_state, and done in memory so that the agent can learn from it
    self.memory.append((state, action, reward, next_state, done)) # Pops the oldest item if we exceed memory


  def train_long_mem(self):
    # Train the agent using the memory
    if len(self.memory) > BATCH_SIZE: # If we have enough items in memory
      mini_sample = random.sample(self.memory, BATCH_SIZE) # Sample a random batch from memory to randomize the training
    else:
      mini_sample = self.memory

    states, actions, rewards, next_states, dones = zip(*mini_sample) # Unzip the sample using pythons zip function
    self.trainer.train_step(states, actions, rewards, next_states, dones) # Train the agent using the sample
    # Could have used a for loop, but this is more efficient


  def train_short_mem(self, state, action, reward, next_state, done):
    # Train the agent using the last state
    self.trainer.train_step(state, action, reward, next_state, done)


  def get_action(self, state):
    # Return the action the agent takes
    # Random moves: tradeoff between exploration/exploitation
    self.epsilon = 80 - self.n_games # The probability of choosing a random action
    final_move = [0, 0, 0] # [straight, right, left]
    if random.randint(0, 200) < self.epsilon: # If a random number between 0 and 200 is less than epsilon
      move = random.randint(0, 2)
      final_move[move] = 1
    # The more games we have the smaller the epsilon will be, so less random actions
    else:
      state0 = torch.tensor(np.array(state), dtype=torch.float) # Convert the state to a tensor, which is a multi-dimensional matrix
      prediction = self.model(state0)
      move = torch.argmax(prediction).item() 
      final_move[move] = 1 # The move with the highest prediction will be the final move

    return final_move
  

    


def train():
  # Train the agent
  plot_scores = []
  plot_mean = [] 
  total_score = 0
  record = 0 # The best score
  agent = Agent()
  game = SnakeGameAI()

  agent.n_games, record, total_score = agent.model.load_checkpoint(agent.optimizer)
  agent.trainer.load_checkpoint()
  
  while True:
    # Old/Current State
    state_old = agent.get_state(game)

    # Get move
    final_move = agent.get_action(state_old)

    # Perform move and get new state
    reward, done, score = game.play_step(final_move) 
    state_new = agent.get_state(game)

    # Train short memory
    agent.train_short_mem(state_old, final_move, reward, state_new, done)
      # The short memory is so that the agent can learn from the last state

    # Remember
    agent.remember(state_old, final_move, reward, state_new, done)
      # Store it all in the deque

    if done: 
      # Train long memory
        # Uses all previous games and the memory to train
      game.reset() # Reset the game using function in game.py
      agent.n_games += 1
      agent.train_long_mem()

      if score > record:
        record = score
        # agent.model.save()
        agent.model.save_checkpoint(agent.optimizer, agent.n_games, record, total_score)
        agent.trainer.save_checkpoint()


      print(f'Game {agent.n_games}, Score: {score}, Record: {record}')  # Print the score

      plot_scores.append(score)  # Append the score to the plot
      total_score += score
      mean_score = total_score / agent.n_games # Calculate the mean score
      plot_mean.append(mean_score)
      plot(plot_scores, plot_mean)
    


if __name__ == '__main__':
  train()