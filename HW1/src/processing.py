#Qiacheng Li
#CS445
#HW1

from perceptron import Perceptron
import random
import csv


class Processing():

    def __init__(self):
        self.perceptron_group = []
        self.scaled_data_inputs = []
        self.target_class = []
        self.number_of_rows = 0
        self.success = 0
        self.total_success = 0
        self.total_rows = 0
        self.l_rate = 0
        self.filename = "test_"
        self.matrix = []
        for i in range(10):
            self.matrix.append([])
            for j in range(10):
                self.matrix[i].append([])
                self.matrix[i][j] = 0

    def apply_learning_algo(self, l_rate):
        self.l_rate = l_rate
        rdweight = []
        for i in range(10):
            for w in range(785):
                rdweight.append(random.uniform(-0.05, 0.05))
            biasWeight = random.uniform(-0.05, 0.05)

            # creating a list of perceptrons, here we need 10 of them
            self.perceptron_group.append(i)
            # initialize perceptron with random weight and the bias of 1
            self.perceptron_group[i] = Perceptron(biasWeight, rdweight, i, 0)
            rdweight = []

        #train the 10 perceptrons for 50 epochs ep = 0 is the epoch 0 value.
        for ep in range(51):
            self.success = 0
            for r in range(self.number_of_rows):
                for i in range(10):
                    self.perceptron_learning(r, l_rate, self.scaled_data_inputs[r], self.target_class[r], ep, self.perceptron_group[i])

                if int(self.target_class[r]) == int(self.predict(self.perceptron_group)):
                    #print("row: "+ str(r) + " tar:" + str(self.target_class[r]) + " pr: " + str(self.predict(self.perceptron_group)))
                    self.success += 1
                    self.total_success += 1
                self.matrix[int(self.target_class[r])][int(self.predict(self.perceptron_group))] += 1

            if ep == 50:
                with open(self.filename + "weights_" + str(self.l_rate) + ".csv", "w+") as outFile:
                    for i1 in range(10):
                        for r1 in range(785):
                            if r1 > 0:
                                outFile.write(str(self.perceptron_group[i1].weight[r1]) + ",")
                            else:
                                outFile.write(str(self.perceptron_group[i1].bias_weight) + ",")
                        outFile.write('\n')


            print("ep" + str(ep) + " " + str(self.success) + "/" + str(self.number_of_rows) + " " + str((int(self.success) / int(self.number_of_rows)) * 100) + "%")
            self.success = 0
            for i in range(10):
                for j in range(10):
                    print(str(self.matrix[i][j]) + ",", end='')
                    self.matrix[i][j] = 0
                    if j == 9:
                        print('\n')


    def predict(self, perceptron_group):
        maxIndex = 0
        for i in range(10):
            if perceptron_group[i].wx > perceptron_group[maxIndex].wx:
                maxIndex = i
        #print(maxIndex)
        return int(maxIndex)


    #applying the perceptron learning rule to update weights.
    def perceptron_learning(self, row, l_rate, x_inputs, target_class, epochs, perceptron):


        wx = 0
        for ep in range(int(epochs)):

            for i in range(785):
                wx += float(perceptron.weight[i]) * int(x_inputs[i])
            wx += perceptron.bias_weight
            perceptron.wx = wx

            if wx >= 0:
                y = 1
            else:
                y = 0

            if int(perceptron.label) == int(target_class):
                t = 1
            else:
                t = 0

            if t == y:
                return

            Ndiff = float(l_rate * (t - y))

            for i in range(785):
                delta_weight = Ndiff * float(x_inputs[i])
                new_weight = perceptron.weight[i] + delta_weight
                perceptron.weight[i] = new_weight

            perceptron.bias_weight += Ndiff





    #reading the mnist test set and scale the data inputs to be either 0/1
    def read_csv(self):
        with open("mnist_test.csv") as data_file:
            reader = csv.reader(data_file)
            mnist_data = []
            rows = 0
            for r in reader:
                mnist_data.append(r)
                self.target_class.append(int(r[0]))
                rows = rows + 1
        self.number_of_rows = rows

        for i in range(rows):
            for j in range(785):
                mnist_data[i][j] = int(int(mnist_data[i][j]) / 255)

        with open("scaled_mnist_test.csv", "w", newline='') as write_file:
            writer = csv.writer(write_file, quoting=csv.QUOTE_ALL)
            for r in mnist_data:
                writer.writerow(r)
        """
        for x in range(rows):
            print(self.target_values[x])
        """

        write_file.close()

    #loading scaled test/training data
    def load_data(self):
        with open("scaled_mnist_test.csv") as data_file:
            reader = csv.reader(data_file)
            rows = 0
            for r in reader:
                self.scaled_data_inputs.append(r)

                rows = rows + 1
        self.number_of_rows = rows

