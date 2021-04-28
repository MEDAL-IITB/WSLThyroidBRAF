import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.L = 250
        self.D = 64
        self.K = 1

        # self.feature_extractor_part1 = nn.Sequential(
        #     nn.Conv2d(3, 8, kernel_size=5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(6, stride=4),
        #     nn.Conv2d(8, 12, kernel_size=3),
        #     nn.ReLU(),
        #     nn.MaxPool2d(4, stride=3)
        # )

        self.feature_extractor_part2 = nn.Sequential(
            #nn.Linear(50 * 4 * 4, self.L),
	    nn.Linear(1*512, self.L),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # print(x.size())
        x = x.squeeze(0)
        # print(x.size())
        # H = self.feature_extractor_part1(x)
        #H = H.view(-1, 50 * 4 * 4)
        H = x.view(-1, 1*512)
        # print(H.size())
        H = self.feature_extractor_part2(H)  # NxL
        # print(H.size())
        # print(torch.Size(H))
        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        #rint type(Y)
        #print type(X)
        Y = Y.float()
        X = X.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data[0]
        return error, Y_hat

    def calculate_objective(self, X, Y):
        #print type(Y.data)
        #print type(X.data)
        Y = Y.float()
        X = X.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A
