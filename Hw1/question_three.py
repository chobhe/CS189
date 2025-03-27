import utils
from scipy import io
import numpy as np
import sklearn.svm as svm
import sklearn.metrics as metrics
import matplotlib.pyplot as plt


def question_three():
    hyperparameters = np.logspace(-8, -1, num=8)
    training_data_set_size = 10000
    mnist_data = io.loadmat("Hw1/data/%s_data.mat" % "mnist")


    mnist_shuffled_labels, mnist_shuffled_values = utils.shuffle(mnist_data["training_labels"], mnist_data["training_data"])
    mnist_training_set = np.array(mnist_shuffled_values[:training_data_set_size])
    mnist_training_set_labels = np.array(mnist_shuffled_labels[:training_data_set_size]).ravel()
    mnist_validation_set= np.array(mnist_shuffled_values[training_data_set_size:])
    mnist_validation_set_labels = np.array(mnist_shuffled_labels[training_data_set_size:]).ravel()

    print("mnist training set shape: ", mnist_training_set.shape)
    print("mnist training set labels shape: ", mnist_training_set_labels.shape)
    print("mnist validation set shape: ", mnist_validation_set.shape)
    print("mnist validation set labels shape: ", mnist_validation_set_labels.shape)



    map_hyperparameter_to_accuracy = {}
    for hyperparameter in hyperparameters:
        print("Hyperparameter: ", hyperparameter)
        classifier = svm.SVC(C=hyperparameter, kernel='linear', decision_function_shape='ovr').fit(mnist_training_set, mnist_training_set_labels)
        validation_set_classifications = classifier.predict(mnist_validation_set)
        accuracy = metrics.accuracy_score(mnist_validation_set_labels, validation_set_classifications)
        print("Accuracy: ", accuracy)
        map_hyperparameter_to_accuracy[hyperparameter] = accuracy

    print("Hyperparameter to Accuracy: ", map_hyperparameter_to_accuracy)

    # plot the accuracies with a logarithmic x-axis
    plt.plot(hyperparameters, list(map_hyperparameter_to_accuracy.values()), marker='o')
    plt.xscale('log')  # Set x-axis to logarithmic scale
    plt.xlabel("Hyperparameter (C)")
    plt.ylabel("Accuracy")
    plt.title("Hyperparameter (C) vs Accuracy")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)  # Optional: Add grid for better readability
    plt.show()


question_three()