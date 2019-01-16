

class Perceptron():

    def __init__(self, bias, weight, inputs):
        self.bias = bias
        self.weight = weight


    # l is the learning rate
    # t is the target output
    # y is the predicted output
    # x is the input
    def perceptron_learning(self, l, target, x_input):
        print("Learning..")
        for x in range(785):
            activation = self.weight * x_input[x]

        print(activation)
        activation += self.bias

        if activation >= 0:
            prediction = 1
        else:
            prediction = 0
        delta_weight = l * (target - prediction) * input[x]
        print(delta_weight)
        self.weight = self.weight + delta_weight
