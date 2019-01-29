#Qiacheng Li
#CS445
#HW1
from processing import Processing


def main():

    process = Processing()

    process.read_csv()
    process.load_data()
    process.apply_learning_algo(0.1)




if __name__ == "__main__":
    main()
