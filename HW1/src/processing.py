from perceptron import Perceptron
import random
import csv


class Processing():

    def __init__(self):
        self.perceptron_group = []
        self.scaled_data_inputs = []
        self.target_values = []
        self.number_of_rows = 0
        self.success = 0

    def apply_learning_algo(self, l_rate):
        rdweight = []
        for x in range(10):
            for w in range(785):
                rdweight.append(random.uniform(-0.05, 0.05))

            # creating a list of perceptrons, here we need 10 of them
            self.perceptron_group.append(x)
            # initialize perceptron with random weight and the bias of 1
            self.perceptron_group[x] = Perceptron(1, rdweight, x)
        print(self.number_of_rows)
        for x in range(self.number_of_rows):
            for i in range(10):
                self.perceptron_learning(x, l_rate, self.scaled_data_inputs[x], self.perceptron_group[i])


    # l is the learning rate
    # t is the target output
    # y is the predicted output
    # x is the input
    def perceptron_learning(self, row, l_rate, x_inputs, perceptron):
        wx = 0
        for x in range(785):
            wx += float(perceptron.weight[x]) * float(x_inputs[x])


            if wx >= 0:
                prediction = 1
            else:
                prediction = 0

            if perceptron.label == self.target_values[row]:
                target = 1
            else:
                target = 0

            delta_weight = float(l_rate * (target - prediction)) * float(x_inputs[x])
            new_weight = perceptron.weight[x] + delta_weight

            if perceptron.label == self.target_values[x] and target == prediction:
                self.success += 1
                #print(perceptron.label)


                #print("label: " + str(perceptron.label) + "target: " + str(self.target_values[x]), str(target), str(prediction))



        #if(perceptron.weight[x] != new_weight):
         #   print("wx: " + str(wx) + " delta weight: " + str(delta_weight) + " new weight: " + str(new_weight) + " target: " + str(target)
          #      + " prediction: " + str(prediction))
        return new_weight


    def read_csv(self):
        with open("mnist_train.csv") as data_file:
            reader = csv.reader(data_file)
            mnist_data = []
            rows = 0
            for r in reader:
                mnist_data.append(r)
                self.target_values.append(int(r[0]))

        for i in range(rows):
            for j in range(785):
                mnist_data[i][j] = int(int(mnist_data[i][j]) / 255)

        with open("scaled_mnist_train.csv", "w", newline='') as write_file:
            writer = csv.writer(write_file, quoting=csv.QUOTE_ALL)
            for r in mnist_data:
                writer.writerow(r)
        """
        for x in range(rows):
            print(self.target_values[x])
        """

        write_file.close()

    def load_data(self):
        with open("scaled_mnist_train.csv") as data_file:
            reader = csv.reader(data_file)
            rows = 0
            for r in reader:
                self.scaled_data_inputs.append(r)

                rows = rows + 1
        self.number_of_rows = rows

