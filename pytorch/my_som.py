# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Štefan Pócoš, Iveta Bečková 2017-2023
import math

import matplotlib.pyplot as plt
import numpy as np

def L_max(i, j, axis=0):
    return np.max(np.abs(i - j), axis=axis)
def L_1(i, j):
    return np.sum(np.abs(i - j))

def L_2(i, j):
    return np.sqrt(np.sum(np.power(i - j, 2)))

class SOM():

    def __init__(self, dim_in, n_rows, n_cols, inputs=None):
        self.dim_in = dim_in
        self.n_rows = n_rows
        self.n_cols = n_cols

        self.weights = np.random.rand(self.n_rows, self.n_cols, self.dim_in) # FIXED (we will solve this together on the exercises):

        if inputs is not None:  # FIXED (we will solve this together on the exercises):
            # "Fill" the input space with neurons - scale and shift neurons to inputs' distribution.
            # Note: SOM will train even without it, but it helps.

            min_input = np.min(inputs, axis=1)
            max_input = np.max(inputs, axis=1)

            self.weights *= (max_input - min_input)
            self.weights += min_input


    def euclid_dist(self, x, y):
        return np.sum((x-y)**2)

    def winner(self, x):
        '''
        Find winner neuron and return its coordinates in grid (i.e. its "index").
        Iterate over all neurons and find the neuron with the lowest distance to input x (np.linalg.norm).
        '''

        # FIXED
        win_r, win_c = -1, -1
        best_dist = math.inf
        for r in range(self.weights.shape[0]):
            for c in range(self.weights.shape[1]):
                act_dist = self.euclid_dist(x, self.weights[r][c])
                if act_dist < best_dist:
                    best_dist = act_dist
                    win_c, win_r = c, r

        return win_r, win_c


    def train(self,
              inputs,   # Matrix of inputs - each column is one input vector
              eps=100,  # Number of epochs
              alpha_s=0.01, alpha_f=0.001, lambda_s=None, lambda_f=1,  # Start & end values for alpha & lambda
              discrete_neighborhood=True,  # Use discrete or continuous (gaussian) neighborhood function
              grid_metric=(lambda u,v:0),  # Grid distance metric
              live_plot=False, live_plot_interval=10,  # Draw plots dring training process
              logs = False
             ):

        (_, count) = inputs.shape
        plot_in3d = self.dim_in > 2

        quant_errors = []
        average_adjustments = []
        lambdas = []
        alphas = []

        for ep in range(eps):
            alpha_t  = alpha_s * (alpha_f/alpha_s)**(ep/(eps-1))  # FIXED
            lambda_t = lambda_s * (lambda_f/lambda_s)**(ep/(eps-1))  # FIXED

            cur_adjustment = 0

            for idx in np.random.permutation(count):
                x = inputs[:, idx]

                win_r, win_c = self.winner(x)

                # Use "d = grid_metric(vector_a, vector_b)" for grid distance
                # Use discrete neighborhood
                if discrete_neighborhood:
                    for r in range(self.n_rows):
                        for c in range(self.n_cols):
                            d = grid_metric(np.array([win_r, win_c]), np.array([r, c]))
                            h_t = 1 if d < lambda_t else 0
                            self.weights[r, c] += alpha_t*(x - self.weights[r, c]) * h_t  # FIXED
                            cur_adjustment += np.sum(np.abs(alpha_t*(x - self.weights[r, c]) * h_t))
                else:
                    for r in range(self.n_rows):
                        for c in range(self.n_cols):
                            d = grid_metric(np.array([win_r, win_c]), np.array([r, c]))
                            h_t = np.exp(-np.power(d, 2)/np.power(lambda_t, 2))
                            self.weights[r, c] += alpha_t * (x - self.weights[r, c]) * h_t
                            cur_adjustment += np.sum(np.abs(alpha_t*(x - self.weights[r, c]) * h_t))

            # count error
            quant_error = 0
            for idx in np.random.permutation(count):
                x = inputs[:, idx]
                win_r, win_c = self.winner(x)
                quant_error += np.min(np.abs(x - self.weights[win_r, win_c]))
            quant_error /= count
            quant_errors.append(quant_error)

            # adjustment
            cur_adjustment /= count * self.n_cols * self.n_rows
            average_adjustments.append(cur_adjustment/100)

            # lambdas, alphas
            lambdas.append(lambda_t / 1000)
            alphas.append(alpha_t / 100)

            if logs:
                print('Ep {:3d}/{:3d}:  alpha_t = {:.3f}, lambda_t = {:.3f}, quant err = {:.5f}'
                  .format(ep+1, eps, alpha_t, lambda_t, quant_error))

        return  quant_errors, average_adjustments, alphas, lambdas


    def neuron_classes_diagram(self, rows, cols, inputs, classes):

        freqs = [[{} for _ in range(cols)] for _ in range(rows)]

        (_, count) = inputs.shape
        for idx in np.random.permutation(count):
            x = inputs[:, idx]
            win_r, win_c = self.winner(x)

            act_class = classes[idx]

            if act_class not  in freqs[win_r][win_c]:
                freqs[win_r][win_c][act_class] = 0
            freqs[win_r][win_c][act_class] += 1

        neuron_classes = []
        percentage = []
        x = []
        y = []
        for r in range(rows):
            for c in range(cols):
                if freqs[r][c] == {}: continue
                neuron_classes.append(max(freqs[r][c], key=freqs[r][c].get))
                percentage.append(neuron_classes[-1] / sum(freqs[r][c].values()) * 100)
                x.append(c)
                y.append(r)


        # print(neuron_classes)
        plt.figure()
        c = neuron_classes
        s = percentage
        plt.rc('axes', axisbelow=True)
        plt.grid(linestyle='dashed')
        scatter = plt.scatter(x, y, c=c, s=s)
        plt.legend(*scatter.legend_elements(),
                    loc="lower left", title="Classes")
        #plt.imshow(neuron_classes, cmap='viridis', interpolation='nearest')
        #plt.colorbar()
        plt.title('Neuron activation classes')

    def attr_value_heatmap(self, rows, cols):

        fig, axs = plt.subplots(nrows=3, ncols=3, gridspec_kw={'hspace': 0.7})
        axs = axs.reshape(9)

        for attr in range(7):
            act_weights = self.weights[:, :, attr]
            axs[attr].imshow(act_weights)
            axs[attr].title.set_text(f'{attr+1}. atribute')

    def u_matrices(self, rows, cols):

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, gridspec_kw={'hspace': 0.7})


        dist_horiz = []
        for i in range(0, rows):
            tmp = []
            for j in range(1, cols):
                tmp.append(self.euclid_dist(self.weights[i, j-1], self.weights[i, j]))
            dist_horiz.append(tmp.copy())

        dist_vert = []
        for i in range(1, rows):
            tmp = []
            for j in range(0, cols):
                tmp.append(self.euclid_dist(self.weights[i-1, j], self.weights[i, j]))
            dist_vert.append(tmp.copy())

        ax1.imshow(dist_horiz, cmap='gray', interpolation='nearest')
        ax1.title.set_text('Horizontal distances')
        ax2.imshow(dist_vert, cmap='gray', interpolation='nearest')
        ax2.title.set_text('Vertical distances')


    def plot_training(self, quant_errors, average_adjustments, alphas, lambdas):
        # plot error and adjustment
        plt.figure()
        plt.plot(quant_errors, label="Quantization error")
        plt.plot(average_adjustments, label="Average adjustment / 100")
        plt.plot(alphas, label="Alpha decay / 100")
        plt.plot(lambdas, label="Lambda decay / 1000")
        plt.legend()