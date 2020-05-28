import mnist_loader
import network as n

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = n.Network([784, 30, 10], cost=n.CrossEntropyCost)
net.stochastic_gradient_decent(training_data, 30, 10, 0.5, validation_data, regularisation_parameter = 5.0, output_report=True)
