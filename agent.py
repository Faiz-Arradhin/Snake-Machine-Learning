import torch
import random
import numpy as np
from collections import deque
from game import SnakeGame, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
import cv2


grid_size2 = 9
cell_size2 = 50
# Calculate the size of the Sudoku grid
image_size2 = grid_size2 * cell_size2
image2 = np.ones((image_size2 + cell_size2+75, image_size2+800), np.uint8) * 255
cv2.namedWindow("Snake")
cv2.setWindowProperty("Snake", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


image2.fill(0)
tulisan="Snake Machine Learning" 
cv2.putText(image2,tulisan, (10,80), cv2.FONT_HERSHEY_SIMPLEX, 2.3, (255,255,255), 1)
tulisan="How to Use:" 
cv2.putText(image2,tulisan, (10,150), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255,255,255), 1)
tulisan="  1. Observe how the snake learn what it should and shouldn't do" 
cv2.putText(image2,tulisan, (10,210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
tulisan="  2. Improvement will be seen significantly at about the 100th try" 
cv2.putText(image2,tulisan, (10,240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

tulisan="Click any alphabet key on the keyboard to start" 
cv2.putText(image2,tulisan, (10,420), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 1)
cv2.imshow('Snake', image2)
cv2.waitKey(0)



MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.Left
        dir_r = game.direction == Direction.Right
        dir_u = game.direction == Direction.Up
        dir_d = game.direction == Direction.Down

        state = [
            # Danger straight
            (dir_r and game.collision(point_r)) or 
            (dir_l and game.collision(point_l)) or 
            (dir_u and game.collision(point_u)) or 
            (dir_d and game.collision(point_d)),

            # Danger right
            (dir_u and game.collision(point_r)) or 
            (dir_d and game.collision(point_l)) or 
            (dir_l and game.collision(point_u)) or 
            (dir_r and game.collision(point_d)),

            # Danger left
            (dir_d and game.collision(point_r)) or 
            (dir_u and game.collision(point_l)) or 
            (dir_r and game.collision(point_u)) or 
            (dir_l and game.collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        
        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGame()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()