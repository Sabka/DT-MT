import torch
import torch.nn as nn
import numpy as np

class SOM(nn.Module):
    """
    2-D Self-Organizing Map with Gaussian Neighbourhood function
    and linearly decreasing learning rate.
    """

    def __init__(self, m, n, dim, niter, args, alpha=None, sigma=None):
        super(SOM, self).__init__()
        self.args = args
        self.m = m
        self.n = n
        self.dim = dim
        self.niter = niter
        if alpha is None:
            self.alpha = 0.3
        else:
            self.alpha = float(alpha)
        if sigma is None:
            self.sigma = max(m, n) / 2.0
        else:
            self.sigma = float(sigma)

        self.weights = torch.randn(m * n, dim).to(self.args.device)
        self.locations = torch.LongTensor(np.array(list(self.neuron_locations())))
        self.pdist = nn.PairwiseDistance(p=2)
        self.quant_err = 0
        self.num_err = 0
        self.winner_occurences = [0 for i in range(self.m*self.n)]

    def get_weights(self):
        return self.weights

    def get_locations(self):
        return self.locations
    def get_som_stats(self):

        quant_err = self.quant_err/self.num_err
        winner_discrimination = sum(tmp > 0 for tmp in self.winner_occurences) / self.num_err

        px = torch.FloatTensor([x/self.num_err for x in self.winner_occurences])
        logpx = torch.log(px)
        product = px * logpx
        entropy = - torch.sum(product)
        
        self.quant_err = 0
        self.num_err = 0

        return quant_err, winner_discrimination, entropy

    def neuron_locations(self):
        for i in range(self.m):
            for j in range(self.n):
                yield np.array([i, j])

    def map_vects(self, input_vects):
        to_return = []
        for vect in input_vects:
            min_index = min([i for i in range(len(self.weights))],
                            key=lambda x: torch.norm(vect - self.weights[x]))
            to_return.append(self.locations[min_index])

        return to_return

    def bmu_loc(self, x):
        
        #print(self.weights.get_device())
        x_matrix = torch.stack([x for i  in range(self.m * self.n)]).to(self.args.device)
        #print(x_matrix.get_device())
        dists = self.pdist(x_matrix, self.weights)
        _, bmu_index = torch.min(dists, 0)
        bmu_loc = self.locations[bmu_index, :]
        bmu_loc = bmu_loc.squeeze()

        return bmu_loc, bmu_loc[0] * self.m + bmu_loc[1]

    def forward(self, x, it):

        """ Take one input x and find location of its BMU in 2D net,
            then calculate distances of all neurons to this BMU and
            update their positions. """

        bmu_loc, bmu_loc_1D = self.bmu_loc(x)

        # calculate distance of bmu position in ND space and x
        winner = self.weights[bmu_loc_1D]
        self.quant_err += torch.sum(torch.pow(x - winner, 2))
        self.num_err += 1

        # calculate winner occurences for stats
        self.winner_occurences[bmu_loc_1D] += 1

        learning_rate_op = 1.0 - it / self.niter
        alpha_op = self.alpha * learning_rate_op
        sigma_op = self.sigma * learning_rate_op

        tmp = torch.stack([bmu_loc for i in range(self.m * self.n)])
        tmp = self.locations.float() - tmp.float()
        tmp = torch.pow(tmp, 2)
        bmu_distance_squares = torch.sum(tmp, 1)


        #bmu_distance_squares = torch.sum(
        #    torch.pow(self.locations.float() - torch.stack([bmu_loc for i in range(self.m * self.n)]).float(), 2), 1)

        neighbourhood_func = torch.exp(torch.neg(torch.div(bmu_distance_squares, sigma_op ** 2)))

        learning_rate_op = alpha_op * neighbourhood_func

        learning_rate_multiplier = torch.stack(
            [learning_rate_op[i:i + 1].repeat(self.dim) for i in range(self.m * self.n)])
        delta = torch.mul(learning_rate_multiplier.to(self.args.device), (torch.stack([x for i in range(self.m * self.n)]).to(self.args.device) - self.weights))
        new_weights = torch.add(self.weights, delta)
        self.weights = new_weights

