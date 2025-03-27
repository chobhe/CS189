import utils
from scipy import io
import numpy as np
import sklearn.svm as svm
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

def part_a():
    # Note the svm is primarily a binary classifier
    # we can do a multi class classification by training a binary classifier for each class against all of the others COMBINED (one vs all or OVR) 
    # So we would train a classifier for the number 0 against 1-9 combined, then 1 against 0,2-9 combined, etc.
    # We can also do one vs one (OVO) where we train a classifier for each pair of classes.
    training_example_counts = [100, 200, 500, 1000, 2000, 5000, 10000]
    mnist_data = io.loadmat("Hw1/data/%s_data.mat" % "mnist")

    map_count_to_accuracy = {}

    for count in training_example_counts:   
        mnist_shuffled_labels, mnist_shuffled_values = utils.shuffle(mnist_data["training_labels"], mnist_data["training_data"])
        mnist_training_set = np.array(mnist_shuffled_values[:count])
        mnist_training_set_labels = np.array(mnist_shuffled_labels[:count]).ravel()
        mnist_validation_set= np.array(mnist_shuffled_values[count:])
        mnist_validation_set_labels = np.array(mnist_shuffled_labels[count:]).ravel()

        # Create a classifier: a support vector classifier
        classifier = svm.SVC(kernel='linear', decision_function_shape='ovr').fit(mnist_training_set, mnist_training_set_labels.ravel())
        validation_set_classifications = classifier.predict(mnist_validation_set)

        accuracy = metrics.accuracy_score(mnist_validation_set_labels, validation_set_classifications)
        map_count_to_accuracy[count] = accuracy


    # plot the accuracies
    plt.plot(list(map_count_to_accuracy.keys()), list(map_count_to_accuracy.values()))
    plt.xlabel("Number of Training Examples")
    plt.ylabel("Accuracy")
    plt.title("Number of Training Examples vs Accuracy")
    plt.show()
    return 



def part_b():
    spam_data = io.loadmat("Hw1/data/%s_data.mat" % "spam")
    training_example_counts = [100, 200, 500, 1000, 2000]

    print("training_example_counts: " + str(training_example_counts))

    map_count_to_accuracy = {}

    for count in training_example_counts:   
        spam_shuffled_labels, spam_shuffled_values = utils.shuffle(spam_data["training_labels"], spam_data["training_data"])
        spam_training_set = np.array(spam_shuffled_values[:count])
        spam_training_set_labels = np.array(spam_shuffled_labels[:count]).ravel()
        spam_validation_set= np.array(spam_shuffled_values[count:])
        spam_validation_set_labels = np.array(spam_shuffled_labels[count:]).ravel()


        # Create a classifier: a support vector classifier
        classifier = svm.SVC(kernel='linear', decision_function_shape='ovr').fit(spam_training_set, spam_training_set_labels.ravel())
        validation_set_classifications = classifier.predict(spam_validation_set)

        accuracy = metrics.accuracy_score(spam_validation_set_labels, validation_set_classifications)
        map_count_to_accuracy[count] = accuracy


     # plot the accuracies
    plt.plot(list(map_count_to_accuracy.keys()), list(map_count_to_accuracy.values()))
    plt.xlabel("Number of Training Examples")
    plt.ylabel("Accuracy")
    plt.title("Number of Training Examples vs Accuracy")
    plt.show()


    return 

def part_c():
    cifar_data = io.loadmat("Hw1/data/%s_data.mat" % "cifar10")
    training_example_counts = [100, 200, 500, 1000, 2000,5000]

    print("training_example_counts: " + str(training_example_counts))

    map_count_to_accuracy = {}

    for count in training_example_counts:   
        cifar_shuffled_labels, cifar_shuffled_values = utils.shuffle(cifar_data["training_labels"], cifar_data["training_data"])
        cifar_training_set = np.array(cifar_shuffled_values[:count])
        cifar_training_set_labels = np.array(cifar_shuffled_labels[:count]).ravel()
        cifar_validation_set= np.array(cifar_shuffled_values[count:])
        cifar_validation_set_labels = np.array(cifar_shuffled_labels[count:]).ravel()


        # Create a classifier: a support vector classifier
        classifier = svm.SVC(kernel='linear', decision_function_shape='ovr').fit(cifar_training_set, cifar_training_set_labels.ravel())
        validation_set_classifications = classifier.predict(cifar_validation_set)

        accuracy = metrics.accuracy_score(cifar_validation_set_labels, validation_set_classifications)
        map_count_to_accuracy[count] = accuracy


     # plot the accuracies
    plt.plot(list(map_count_to_accuracy.keys()), list(map_count_to_accuracy.values()))
    plt.xlabel("Number of Training Examples")
    plt.ylabel("Accuracy")
    plt.title("Number of Training Examples vs Accuracy")
    plt.show()


part_c()