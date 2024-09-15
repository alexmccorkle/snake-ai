import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple # This is a tuple with named fields
# Tuple ELI5: It's like a list, but you can't change the values inside it.

pygame.init()
font = pygame.font.Font('arial.ttf', 25)

# Reset function for agent to be able to reset the game and start again

# Reward function to give the agent a reward for eating the food and a penalty for dying

# Play(action) function to play the game based on the action the agent takes

# Game iteration

# is_collision



class Direction(Enum): 
# This is super useful for defining constants because it makes it easier to read the code and less error-prone
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# Constants
SPEED = 20
BLOCKSIZE = 20

## Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BLUE2 = (0, 100, 255)

class SnakeGameAI:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h


        # initialize display: 
        self.display = pygame.display.set_mode((self.w, self.h)) # width, height
        pygame.display.set_caption('Snake Game')
        self.clock = pygame.time.Clock() # To control speed of game
        self.reset()
        
        


    def reset(self):
        # init game state:
        self.direction = Direction.RIGHT

        # Snake position
        self.head = Point(self.w/2, self.h/2)
        # Snake body
        self.snake = [self.head, Point(self.head.x-BLOCKSIZE, self.head.y), Point(self.head.x-(2*BLOCKSIZE), self.head.y)]
        
        self.score = 0
        self.food = None

        # Place food
        self._place_food()
        self.frame_iteration = 0


    def _place_food(self): # This is a private method or a helper method
        x = random.randint(0, (self.w-BLOCKSIZE)//BLOCKSIZE )*BLOCKSIZE # // is integer division
        y = random.randint(0, (self.h-BLOCKSIZE)//BLOCKSIZE )*BLOCKSIZE # Ensures that the food is placed in a multiple of BLOCKSIZE
        self.food = Point(x, y)

        if self.food in self.snake:
            self._place_food() # Recursion to place the food again if it's placed on the snake

        

    def play_step(self, action):
        self.frame_iteration += 1 # To keep track of how many frames have passed
        # 1. User Input
        for event in pygame.event.get() : # Get's the user input
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()


        # 2. Move Snake
        self._move(action) # Update the head
        self.snake.insert(0, self.head) # Inserting the head of the snake at the beginning of the list

        # 3. Check if Game Over
        reward = 0
        game_over = False
        if self._is_collision() or self.frame_iteration > 100*len(self.snake): # So that the game ends if the AI doesn't do anything
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. Place new food if we hit food, or just move snake
        if self.head == self.food: # If the head of the snake is at the same position as the food
            self.score += 1
            reward = 10
            self._place_food() # Place new food at a random location
        else: 
            self.snake.pop() # Remove the last element of the snake so that it is not growing infinitely

        # 5. Update display and clock
        self._update_ui()
        self.clock.tick(SPEED) # SPEED frames per second

        # 6. Return game over and score
        return reward, game_over, self.score

    def _update_ui(self):
        self.display.fill(BLACK) # First fill screen before drawing anything else
        # Draw Snake
        for pt in self.snake: # pt = point and we made point a named tuple
            pygame.draw.rect(self.display, BLUE, pygame.Rect(pt.x, pt.y, BLOCKSIZE, BLOCKSIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        # Draw Food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCKSIZE, BLOCKSIZE))

        # Score 
        text = font.render('Score: ' + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0]) # Upper left corner
        pygame.display.flip() # Update the display


    def _move(self, action):
        # [straight, right, left] in the format [0, 1, 0] = right

        clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clockwise.index(self.direction) # Get the index of the current direction

        if np.array_equal(action, [1, 0, 0]):
            new_direction = clockwise[idx] # No change in direction
        elif np.array_equal(action, [0, 1, 0]): # right turn = right -> down -> left -> up
            next_idx = (idx + 1) % 4 # To make sure that the index doesn't go out of bounds and stays within 0, 1, 2, 3
            new_direction = clockwise[(idx)] # Turn right
        else: # [0, 0, 1] # left turn = right -> up -> left -> down
            next_idx = (idx - 1) % 4 # Same as above but for turning left
            new_direction = clockwise[next_idx]

        self.direction = new_direction


        x = self.head.x 
        y = self.head.y

        if self.direction == Direction.RIGHT:
            x += BLOCKSIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCKSIZE
        elif self.direction == Direction.DOWN:
            y += BLOCKSIZE
        elif self.direction == Direction.UP:
            y -= BLOCKSIZE

        self.head = Point(x, y)

    def _is_collision(self, pt=None): # pt is the point we want to check for collision
        # If it hits the border
        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCKSIZE or pt.x < 0 or pt.y > self.h - BLOCKSIZE or pt.y < 0:
            return True
        # If it hits itself
        if pt in self.snake[1:] : # We don't want to check the head because it's the first element
            return True
        
        return False
