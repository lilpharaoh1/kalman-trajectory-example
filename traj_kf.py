import numpy as np

class KF:
    def __init__(self):
        self.counter = 0

    def add_one(self):
        self.counter += 1
        return 1

    def print_counter(self):
        print(self.counter)