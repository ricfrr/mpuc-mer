import torch
import numpy as np 
class ModelMonitoring(object):
    """ 
        Model Monitoring 
    """

    def __init__(self, patience: int,score_delta: float = 0.0):

        if patience < 1:
            raise ValueError("Argument patience should be positive integer.")

        if score_delta < 0.0:
            raise ValueError("Argument score_delta should not be a negative number.")

        self.patience = patience
        self.score_delta = score_delta
        self.stopped = False
        self.counter = 0
        self.best_score = None

    def __call__(self, score: float):

        if np.isnan(score):
            raise RuntimeError(f"Evaluating models failed. Received eval results {score}")

        if self.best_score is None:
            self.best_score = score
        elif score + self.score_delta <= self.best_score:
            if score > self.best_score:
                self.best_score = score
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped = True
        else:
            self.best_score = score
            self.counter = 0


