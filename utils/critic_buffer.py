import torch
import torch.nn as nn
import torch.nn.functional as F

'''saves previous values of scaled_attend_logits'''
class CriticBuffer():
    def __init__(self, attend_heads=4):
        self.prev_attend = [None for _ in range(attend_heads)]  # previous attention values

    def get_prev_attend(self, i_head, new_attend):
        if self.prev_attend[i_head] is None:
            self.prev_attend[i_head] = new_attend
            return None

        prev_attend = self.prev_attend[i_head]
        self.prev_attend = 0.2 * prev_attend + 0.8 * new_attend

        return prev_attend

    '''
    def update_attend_weights(self, i_head, new_attend):
        if self.prev_attend[i_head] is None:        # for the init state
            self.prev_attend[i_head] = new_attend
            return new_attend

        new_logits = 0.2 * self.prev_attend[i_head] + 0.8 * new_attend
        self.prev_attend[i_head] = new_logits

        return new_logits
    '''




        



