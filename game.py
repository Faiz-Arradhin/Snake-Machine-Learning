import cv2
import numpy as np
import random
from enum import Enum
from collections import namedtuple

Point = namedtuple("Point", "x y")
Size = 10

class Direction(Enum):
    Up = 1
    Right = 2
    Down = 3
    Left = 4

class SnakeGame:
    def __init__(self, w=1000, h=680):
        self.w = w
        self.h = h
        self.background = np.zeros((self.h, self.w, 3), np.uint8)
        cv2.imshow("Snake", self.background)
        cv2.waitKey(1)
        self.reset()

    def reset(self):
        self.direction = Direction.Right
        self.head = Point(490, 340)
        self.snake = [self.head, Point(self.head.x - Size, self.head.y), Point(self.head.x - 2 * Size, self.head.y)]
        self.score = 0
        self.food = None
        self._placefood()
        self.iteration = 0

    def _placefood(self):
        foodX = random.randint(0, (self.w - Size) // Size) * Size
        foodY = random.randint(0, (self.h - Size) // Size) * Size
        self.food = Point(foodX, foodY)
        if self.food in self.snake:
            self._placefood()

    def play(self, action):
        self.iteration += 1
        self.tail = self.snake[-1]
        self.dx = self.tail.x - self.snake[-2].x
        self.dy = self.tail.y - self.snake[-2].y
        self.new_segment = Point(self.tail.x + self.dx, self.tail.y + self.dy)
        self._move(action)
        self.snake.insert(0, self.head)
        reward = 0
        game_over = False
        if self.collision() or self.iteration > 180 * len(self.snake):
            game_over = True
            reward = -10
            if self.tipe==2:
                reward-=10
            return reward, game_over, self.score
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._add_segment(self.new_segment)
            self._placefood()
        else:
            self.snake.pop()
        self._updateUI()
        cv2.waitKey(10)
        return reward, game_over, self.score

    def collision(self, pt=None):
        self.tipe=0
        if pt is None:
            pt = self.head
        if pt.x > self.w - Size or pt.x < 0 or pt.y > self.h - Size or pt.y < 0:
            self.tipe=1
            return True
        if pt in self.snake[1:]:
            self.tipe=2
            return True
        return False

    def _updateUI(self):
        self.background.fill(0)
        for pt in self.snake:
            cv2.rectangle(self.background, (pt.x, pt.y), (pt.x + Size, pt.y + Size), [0, 0, 255], cv2.FILLED)
        cv2.rectangle(self.background, (self.food.x, self.food.y), (self.food.x + Size, self.food.y + Size),
                      [255, 255, 255], cv2.FILLED)
        tulisan = str(self.score)
        cv2.putText(self.background, tulisan, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Snake", self.background)

    def _move(self, action):
        arah = [Direction.Up, Direction.Right, Direction.Down, Direction.Left]
        arah_now = arah.index(self.direction)
        if np.array_equal(action, [1, 0, 0]):
            arah_new = arah[arah_now]
        elif np.array_equal(action, [0, 1, 0]):
            next_dir = (arah_now + 1) % 4
            arah_new = arah[next_dir]
        else:
            next_dir = (arah_now - 1) % 4
            arah_new = arah[next_dir]
        self.direction = arah_new
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.Right:
            x += Size
        elif self.direction == Direction.Left:
            x -= Size
        elif self.direction == Direction.Down:
            y += Size
        elif self.direction == Direction.Up:
            y -= Size

        self.head = Point(x, y)

    def _add_segment(self, baru):
        self.snake.append(baru)
