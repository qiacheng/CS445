#Qiacheng Li
#CS445
#HW2

from preprocessing import PreProcessing
from processing import Processing


import sys



def main():

    preProcessing = PreProcessing("mnist_train.csv")
    #preProcessing.preProcessData()

    # number or hidden units
    processing = Processing(10)
    processing.load_data("mnist_train_scaled.csv", "mnist_train_targetClass.csv")

    processing.processing()


    for arg in sys.argv[1:]:
        print(arg)




if __name__ == "__main__":
    main()