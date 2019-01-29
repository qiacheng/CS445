#Qiacheng Li
#CS445
#HW2

import csv
import os

#This class is intended to preprocess the mnist data set


class PreProcessing(object):

    def __init__(self, csvPath):
        self.csvPath = csvPath



    #scale the data input values to be between 0 and 1 by dividing by 255
    def preProcessData(self):
        mnist_data = []
        target_class = []
        with open(self.csvPath) as data_file:
            reader = csv.reader(data_file)
            rows = 0
            for r in reader:
                mnist_data.append(r)
                target_class.append(r[0])
                rows += 1

        #divide each input data by 255, the first one is the target class we will not use it.
        for i in range(rows):
            for j in range(785):
                if j != 0:
                    mnist_data[i][j] = float(float(mnist_data[i][j]) / 255)
                else:
                    mnist_data[i][j] = 1


        outputPath = str(os.path.splitext(self.csvPath)[0]) + '_' + "scaled.csv"
        with open(outputPath, "w", newline='') as write_file:
            writer = csv.writer(write_file, quoting=csv.QUOTE_ALL)
            for r in mnist_data:
                writer.writerow(r)

        #write the target classes to seperate file for debugging.
        with open(str(os.path.splitext(self.csvPath)[0]) + '_' + "targetClass.csv", "w", newline='') as write_file:
            writer = csv.writer(write_file, quoting=csv.QUOTE_ALL)
            for r in target_class:
                writer.writerow(r)

