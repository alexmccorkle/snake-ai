import pygame
import random
from enum import Enum
from collections import namedtuple # This is a tuple with named fields
# Tuple ELI5: It's like a list, but you can't change the values inside it.

pygame.init()
font = pygame.font.Font('arial.ttf', 25)


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

class SnakeGame:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h


        # initialize display: 
        self.display = pygame.display.set_mode((self.w, self.h)) # width, height
        pygame.display.set_caption('Snake Game')
        self.clock = pygame.time.Clock() # To control speed of game
        


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


    def _place_food(self): # This is a private method or a helper method
        x = random.randint(0, (self.w-BLOCKSIZE)//BLOCKSIZE )*BLOCKSIZE # // is integer division
        y = random.randint(0, (self.h-BLOCKSIZE)//BLOCKSIZE )*BLOCKSIZE # Ensures that the food is placed in a multiple of BLOCKSIZE
        self.food = Point(x, y)

        if self.food in self.snake:
            self._place_food() # Recursion to place the food again if it's placed on the snake

        

    def play_step(self):
        # 1. User Input
        for event in pygame.event.get() : # Get's the user input
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            # Movement
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN
                elif event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT: # Not Else because then any key would be considered right
                    self.direction = Direction.RIGHT

        # 2. Move Snake
        self._move(self.direction) # Update the head
        self.snake.insert(0, self.head) # Inserting the head of the snake at the beginning of the list

        # 3. Check if Game Over
        game_over = False
        if self._is_collision():
            game_over = True
            return game_over, self.score

        # 4. Place new food if we hit food, or just move snake
        if self.head == self.food: # If the head of the snake is at the same position as the food
            self.score += 1
            self._place_food() # Place new food at a random location
        else: 
            self.snake.pop() # Remove the last element of the snake so that it is not growing infinitely

        # 5. Update display and clock
        self._update_ui()
        self.clock.tick(SPEED) # SPEED frames per second

        # 6. Return game over and score
        return game_over, self.score

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


    def _move(self, direction):
        # Extract the values of the head
        x = self.head.x 
        y = self.head.y

        if direction == Direction.RIGHT:
            x += BLOCKSIZE
        elif direction == Direction.LEFT:
            x -= BLOCKSIZE
        elif direction == Direction.DOWN:
            y += BLOCKSIZE
        elif direction == Direction.UP:
            y -= BLOCKSIZE

        self.head = Point(x, y)

    def _is_collision(self):
        # If it hits the border
        if self.head.x > self.w - BLOCKSIZE or self.head.x < 0 or self.head.y > self.h - BLOCKSIZE or self.head.y < 0:
            return True
        # If it hits itself
        if self.head in self.snake[1:] : # We don't want to check the head because it's the first element
            return True
        
        return False

# Main Function
if __name__ == '__main__':
    game = SnakeGame()

    # Game Loop

    while True:
        game_over, score = game.play_step()

        # If game over
        if game_over == True:
            break
        
    print('Final Score', score)
        

    pygame.quit()