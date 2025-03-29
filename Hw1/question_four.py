import utils 
from scipy import io
import numpy as np
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt

def part_four():
    hyperparameters = np.logspace(-6, 2, num=9)
    spam_data = io.loadmat("Hw1/data/spam_data.mat")
    validation_size = spam_data["training_data"].shape[0]//5
    spam_shuffled_labels, spam_shuffled_values = utils.shuffle(spam_data["training_labels"], spam_data["training_data"])

    map_hyperparameter_to_accuracy = {}

    for hyperparameter in hyperparameters:
        accuracy = 0 
        for i in range(5):
            spam_validation_set = np.array(spam_shuffled_values[i*validation_size:(i+1)*validation_size])
            spam_validation_set_labels = np.array(spam_shuffled_labels[i*validation_size:(i+1)*validation_size]).ravel()


            spam_training_set = spam_shuffled_values[0:i*validation_size] + spam_shuffled_values[(i+1)*validation_size:]
            spam_training_set = np.array(spam_training_set)
            spam_training_set_labels = spam_shuffled_labels[0:i*validation_size] + spam_shuffled_labels[(i+1)*validation_size:]
            spam_training_set_labels = np.array(spam_training_set_labels).ravel()


            classifier = svm.SVC(C=hyperparameter, kernel='linear', decision_function_shape='ovr').fit(spam_training_set, spam_training_set_labels)
            spam_validation_set_classifications = classifier.predict(spam_validation_set)

            
            accuracy += metrics.accuracy_score(spam_validation_set_labels, spam_validation_set_classifications)

        map_hyperparameter_to_accuracy[hyperparameter] = accuracy/5

        print("Hyperparameter: ", hyperparameter, "Accuracy: ", accuracy/5)


    print(map_hyperparameter_to_accuracy)

    plt.plot(hyperparameters, list(map_hyperparameter_to_accuracy.values()), marker='o')
    plt.xscale('log')
    plt.xlabel("Hyperparameter (C)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Hyperparameter (C)")
    plt.show()



part_four()
