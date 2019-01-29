
import random
import numpy as np
from layer_unit import  Layer_Unit
import csv


class Processing():

    def __init__(self, n_hidden_units):



        # var to store target classes
        self.target_class = []

        # var to store number of rows/ training samples
        self.number_of_samples = []

        # var to store scaled data inputs
        self.scaled_data_inputs = []

        # target classes
        self.target_class = []

        # Input Layer Units
        self.inputLayer_units = []

        # Hidden Layer Units
        self.hiddenLayer_units = []

        self.number_of_hiddenUnits = n_hidden_units

        self.number_of_outputUnits = 10





    def init_input_to_hidden_layer_weights(self):

        # i to the range of the number of hiddenUnits + 1 size, h0 is 1
        for i in range(785):
            self.inputLayer_units.append([])
            rdWeights = []
            rdWeights.append(0)
            for j in range(1, self.number_of_hiddenUnits + 1):
                    rdWeights.append(random.uniform(-.05, .05))


            self.inputLayer_units[i] = Layer_Unit(rdWeights)

        # 10 weights for each input unit

    def init_hidden_layer_to_output_layer_weights(self):

        for i in range(self.number_of_hiddenUnits + 1):
            rdWeights = []
            for j in range(self.number_of_outputUnits):
                    rdWeights.append(random.uniform(-.05, .05))


            self.hiddenLayer_units.append(Layer_Unit(rdWeights))







    def processing(self):
        self.init_input_to_hidden_layer_weights()
        self.init_hidden_layer_to_output_layer_weights()

        self.input_to_hidden_dot_product(self.scaled_data_inputs[0], self.inputLayer_units)



    def input_to_hidden_dot_product(self, inputLayerinputs, inputLayerUnits):

        WeightsMatrix = []


        for i in range(1, self.number_of_hiddenUnits + 1):
            row = []
            for j in range(785):
                row.append(inputLayerUnits[j].weights[i])
            WeightsMatrix.append(row)

        npMatrix = np.matrix(WeightsMatrix)



        newMatrix = np.dot(npMatrix, inputLayerinputs)

        print(newMatrix)

    def load_data(self, csvPath, targetClassPath):
        rows = 0
        with open(csvPath) as data_file:
            reader = csv.reader(data_file)

            for r in reader:
                self.scaled_data_inputs.append(r)
                rows = rows + 1
        self.number_of_samples = rows

        with open(targetClassPath) as data_file:
            reader = csv.reader(data_file)
            for r in reader:
                self.target_class.append(float(r[0]))




        for i in range(rows):
            for j in range(785):
                self.scaled_data_inputs[i][j] = float(self.scaled_data_inputs[i][j])







