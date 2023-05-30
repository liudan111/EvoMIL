import torch
import torch.nn as nn
import torch.nn.functional as F

""" Copyright © 2018 ilse.maximilian@gmail.com.
    We make some changes on  Attention MIL model to make it suitbale for traning protein embeddings and multi-class classification.
"""
class Attention(nn.Module):
    def __init__(self,classes):
        super(Attention, self).__init__()
        self.L = 800 # in the past, it always 800
        self.D = 128
        self.K = 1

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(80 * 4 * 4, self.L), #80,4,4 in the past 20**8*8, 5 16 16
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, classes),
            # nn.Sigmoid()
        )

    def forward(self, x):
        H = self.feature_extractor_part2(x)  # NxL
        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N
        M = torch.mm(A, H)  # KxL
        logits = self.classifier(M)
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)

        return  Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        # error = 1. - Y_hat.eq(Y).cpu().float().mean().data[0]
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data
        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob,Y_hat, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli
        train_acc = Y_hat.eq(Y).cpu().float().mean().item()
        return neg_log_likelihood, train_acc,A,Y_prob,Y_hat