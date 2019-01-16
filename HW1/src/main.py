from perceptron import Perceptron
import random
import csv


def main():
    perceptron_group = []

    for x in range(10):
        rdweight = random.uniform(-0.05, 0.05)

        # creating a list of perceptrons, here we need 10 of them
        perceptron_group.append(x)
        # initialize perceptron with random weight and the bias of 1
        perceptron_group[x] = Perceptron(1, rdweight, 3)

    #scale_csv()
    load_data()



def scale_csv():
    with open("mnist_train.csv") as data_file:
        reader = csv.reader(data_file)
        mnist_data = []
        rows = 0
        for r in reader:
            mnist_data.append(r)
            rows = rows + 1

    for i in range(rows):
        for j in range(785):
            mnist_data[i][j] = int(int(mnist_data[i][j]) / 255)

    with open("scaled_mnist_train.csv", "w", newline='') as write_file:
        writer = csv.writer(write_file, quoting=csv.QUOTE_ALL)
        for r in mnist_data:
            writer.writerow(r)


    write_file.close()

def load_data():
    with open("scaled_mnist_train.csv") as data_file:
        reader = csv.reader(data_file)
        scaled_data = []
        rows = 0
        for r in reader:
            scaled_data.append(r)
    print(scaled_data[0][0])


if __name__ == "__main__":
    main()