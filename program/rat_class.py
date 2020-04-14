import numpy as np


class Rat:
    def __init__(self):
        self.state = np.random.choice([0,1])
        self.reward = 0

    def step(self, action):
        if self.state == 0:
            if action == 0:
                self.state = 1
                self.reward = 0
            else:
                self.state = 0
                self.reward = 0
        
        else:
            if action == 0:
                self.state = 0
                self.reward = 0
            else:
                self.state = 1
                self.reword = 1

        return self.state, self.reward
