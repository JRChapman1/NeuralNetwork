import numpy as np
import scipy.special as sp
import random
import matplotlib.pyplot as plt
import time
import os
import pathlib as pl
from fpdf import FPDF
import shutil


class PDF(FPDF):

    def __init__(self):
        super().__init__()
        self.alias_nb_pages()
        self.add_page()
        self.set_font('Times', '', 12)

    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(80)
        self.cell(30, 10, 'Network Report', 0, 0, 'C')
        self.ln(20)

    def new_line(self):
        self.cell(ln=1, h=10.0, align='L', w=0, txt='', border=0)

    def write_paragraph(self, text):
        self.write(6, text)
        self.new_line()

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')


class QuadraticCost(object):
    @staticmethod
    def fn(a, y):
        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def delta(z, a, y):
        return (a - y) * sigmoid_derivative(z)

    @staticmethod
    def name():
        return 'quadratic cost'


class CrossEntropyCost(object):
    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(- y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def delta(z, a, y):
        return a - y

    @staticmethod
    def name():
        return 'cross entropy cost'


class Network(object):
    def __init__(self, network_shape, cost=CrossEntropyCost):
        # Set network_shape property, which stores a list of integers specifying the number of neurons in each layer of
        # the network.
        self.network_shape = network_shape
        # Set the num_layers property, which specifies the number of layers in our neural network.
        self.num_layers = len(network_shape)
        # Randomly initialise the weights of the network by generating values from a standard normal distribution. This
        # should be a list of numpy arrays where the (i,j)th entry of the kth array is the weight assigned to the
        # synapse connecting the ith neuron in the (k+1)th layer to the jth neuron in the kth layer of the network.
        self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(network_shape[:-1], network_shape[1:])]
        # Randomly initialise the biases assigned to each neuron from a standard normal distribution. This should be a
        # list of 2 dimensional column vectors where the ith entry of the kth column vector in the list gives the bias
        # assigned to the ith neuron in the kth layer of the network.
        self.biases = [np.random.randn(x, 1) for x in network_shape[1:]]
        self.cost = cost

    def feedforward(self, a):
        # Feed a given input, a, through each layer of the network to obtain its classification of the digit
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def evaluate(self, test_data):
        # Evaluate the success rate of the network by using it to classify our test data
        test_results = [(np.argmax(self.feedforward(x)), y) for x, y in test_data]
        return sum(float(x == y) for x, y in test_results) / len(test_data)

    def evaluate_misclassifications(self, test_data):
        # Evaluate the success rate of the network by using it to classify our test data
        test_results = [(np.argmax(self.feedforward(x)), y) for x, y in test_data]
        distribution_of_misclassifications = [0] * 10
        for x, y in test_results:
            if x != y: distribution_of_misclassifications[y] += 1
        return distribution_of_misclassifications

    def output_misclassifications(self, test_data, save_to=''):
        i = 1
        for x, y in test_data:
            if np.argmax(self.feedforward(x)) != y:
                plt.imshow(x.reshape(28,28), cmap='gray_r')
                if save_to=='':
                    plt.show()
                else:
                    plt.savefig(save_to + str(i) + '_' + str(y) + '.png')
                i += 1

    def plot_graphs(self, test_data, success_rates):
        temp_folder_path = str(pl.Path().absolute()) + '/temp/'
        shutil.rmtree(temp_folder_path, ignore_errors=True)
        pl.Path(temp_folder_path).mkdir(parents=True, exist_ok=True)
        plt.plot(success_rates)
        plt.savefig(temp_folder_path + 'success_rates.png')
        distribution_of_misclassifications = self.evaluate_misclassifications(test_data)
        ind = np.arange(len(distribution_of_misclassifications))  # the x locations for the groups
        width = 0.5  # the width of the bars
        fig, ax = plt.subplots()
        ax.bar(ind, distribution_of_misclassifications, width)
        ax.set_ylabel('Misclassifications of digit')
        ax.set_xticks(ind)
        ax.set_xticklabels(range(0, 10))
        plt.savefig(temp_folder_path + 'distribution_of_misclassifications.png')


    def create_report(self, test_data, training_time, epochs, len_training_data, mini_batch_size, success_rates, learning_rate, regularization_parameter=0, save_misclassifications=False):
        output_folder_name = len([file for file in os.walk(str(pl.Path().absolute()) + "/output")])
        output_folder_path = str(pl.Path().absolute()) + "/output/" + str(output_folder_name)
        pl.Path(output_folder_path).mkdir(parents=True, exist_ok=True)
        self.plot_graphs(test_data, success_rates)
        pdf = PDF()
        pdf.write_paragraph('Network trained with ' + str(epochs) + ' epochs, using ' + str(len_training_data) + ' items of training data partitioned into mini batches of size ' + str(mini_batch_size) + ', in ' + str(round(training_time, 2)) + ' seconds. This achieved a maximum accuracy of ' + str(round(max(success_rates) * 100, 5)) + '% using ' + self.cost.name() + ' with a learning rate of ' + str(learning_rate) + ' and a regularization parameter of ' + str(regularization_parameter) + '.')
        pdf.write_paragraph('The networks classification accuracy after each epoch of training is plotted on the below graph.')
        pdf.image(str(pl.Path().absolute()) + '/temp/success_rates.png', w=120.0, h=80.0, x=30.0)
        pdf.new_line()
        pdf.write_paragraph('Of the ' + str(len(test_data)) + ' items of testing data used, the network misclassified ' + str(sum(self.evaluate_misclassifications(test_data))) + '. The distribution of these misclassifications is shown on the bar chart below.')
        pdf.image(str(pl.Path().absolute()) + '/temp/distribution_of_misclassifications.png', w=120.0, h=80.0, x=30.0)
        pdf.output(output_folder_path + '/network_report.pdf', 'F')
        print('Report saved as: \'' + output_folder_path + '/network_report.pdf\'')

    def stochastic_gradient_decent(self, training_data, epochs, mini_batch_size, learning_rate, test_data,
                                   regularisation_parameter=0, plot_success_rates=False, output_progress=False, output_report=False):
        success_rates = []
        t_start = time.time()
        for i in range(1, epochs + 1):
            # For each epoch (learning iteration) create a partition of the training_inputs. The size of each subset in
            # this partition is specified by the parameter 'mini_batch_size'.
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate, regularisation_parameter)
            success_rates.append(self.evaluate(test_data))
            training_time = time.time() - t_start
            if output_progress:
                print('Epoch {0} Complete. Success Rate: {1}'.format(i, success_rates[-1]))
        print('Query complete in {0} seconds. Best classification accuracy achieved on test data was {1}% using {2} and '
              'a learning rate of {3}.'.format(training_time, max(success_rates) * 100, self.cost.name(),
               learning_rate))
        if plot_success_rates:
            plt.plot(success_rates)
            plt.show()
        if output_report:
            self.create_report(test_data, training_time, epochs, len(training_data), mini_batch_size, success_rates, learning_rate, regularisation_parameter)

    def update_mini_batch(self, mini_batch, learning_rate, regularisation_parameter):
        # For each each mini_batch of training inputs, calculate the partial derivatives of the cost function
        # w.r.t each of the weights and biases in the network by applying the backpropagation algorithm.
        mini_batch_data = np.concatenate([data_item[0] for data_item in mini_batch], axis=1)
        mini_batch_labels = np.concatenate([data_item[1] for data_item in mini_batch], axis=1)
        nabla_w, nabla_b = self.backpropagate(mini_batch_data, mini_batch_labels)
        # Use gradient decent to update the weight and bias vectors in the network by adjusting them in the
        # direction which most rapidly decreases the cost function. These change vectors are then multiplied by
        # a scalar (given by the 'learning_rate' parameter) and are subtracted from the existing weights and
        # biases.
        #self.weights = [(1 - (learning_rate * regularisation_parameter / len(mini_batch))) * w - (learning_rate / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.weights = [(1 - learning_rate * (regularisation_parameter / 50000)) * w - (learning_rate / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (learning_rate / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backpropagate(self, x, y):
        # Make a forward pass through the network to find the weighted input to the neurons in each layer (z^l) and the
        # activations at each layer in the network (z^l).
        activation = x
        activations = [activation]
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            activation = sigmoid(z)
            zs.append(z)
            activations.append(activation)
        # Calculate the partial derivatives of the cost function w.r.t the input to the neurons in the output layer
        # using the equation
        #       delta^L = Nabla_a C (*) sigmoid'(z^L)
        # and work backwards through the network to calculate the partial derivatives of the cost function w.r.t the
        # input to the neurons in all previous layers using the relation
        #       delta^l = ((w^{l+1})^T delta^{l+1}) (*) sigmoid'(z^l)
        # From these partial derivatives, we can calculate the partial derivatives of the cost function w.r.t the
        # weights and biases in the network using the equations
        #       partial C / partial w_{jk}^l = a_k^{l-1} * delta_j^k
        # and
        #       partial C / partial b_j^l = delta_j^l
        delta = self.cost.delta(z, activation, y)
        weight_gradients = [np.dot(delta, activations[-2].transpose())]
        bias_gradients = [delta.sum(axis=1, keepdims=True)]
        for l in range(2, self.num_layers):
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sigmoid_derivative(zs[-l])
            weight_gradients.insert(0, np.dot(delta, activations[-l-1].transpose()))
            bias_gradients.insert(0, delta.sum(axis=1, keepdims=True))
        return weight_gradients, bias_gradients


def sigmoid(x):
    return sp.expit(x)

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))





