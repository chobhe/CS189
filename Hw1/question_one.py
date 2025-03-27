from scipy import io
import utils
import numpy as np



def shuffle_and_split_data():
    mnist_data = io.loadmat("Hw1/data/%s_data.mat" % "mnist")

    # fields = "test_data", "training_data", "training_labels"
    # for field in fields:
    #     print(field, mnist_data[field].shape)
    #     print(mnist_data[field])


    # part a 
    # shuffle the training data
    mnist_shuffled_labels, mnist_shuffled_values = utils.shuffle(mnist_data["training_labels"], mnist_data["training_data"])
    mnist_validation_set_ten_thousand = mnist_shuffled_values[:10000]
    mnist_validation_set_labels = mnist_shuffled_labels[:10000]
    mnist_training_set_fifty_thousand = mnist_shuffled_values[10000:]
    mnist_training_set_labels = mnist_shuffled_labels[1000:]

    print("mnist validation shape: " + str(np.array(mnist_validation_set_ten_thousand).shape))
    print("mnist validation labels shape: " + str(np.array(mnist_validation_set_labels).shape))
    print("mnist training shape: " + str(np.array(mnist_training_set_fifty_thousand).shape))
    print("mnist training labels shape: " + str(np.array(mnist_training_set_labels).shape))



    # # part b
    spam_data = io.loadmat("Hw1/data/spam_data.mat")
    validation_size = spam_data["training_data"].shape[0]//5
    spam_shuffled_labels, spam_shuffled_values = utils.shuffle(spam_data["training_labels"], spam_data["training_data"])
    spam_validation_set_twenty_percent = np.array(spam_shuffled_values[0:validation_size])
    spam_validation_set_labels = np.array(spam_shuffled_labels[0:validation_size])

    spam_training_set_eighty_percent = np.array(spam_shuffled_values[validation_size:])
    spam_training_set_labels = np.array(spam_shuffled_labels[validation_size:])

    print("spam validation: " + str(spam_validation_set_twenty_percent.shape))
    print("spam validation labels: " + str(spam_validation_set_labels.shape))

    print("spam training: " + str(spam_training_set_eighty_percent.shape))
    print("spam training labels: " + str(spam_training_set_labels.shape))


    #  # part c
    # # shuffle the training data
    cifar_data = io.loadmat("Hw1/data/cifar10_data.mat")
    cifar_shuffled_labels, cifar_shuffled_values = utils.shuffle(cifar_data["training_labels"], cifar_data["training_data"])

    cifar_validation_set_five_thousand =  np.array(cifar_shuffled_values[:5000])
    cifar_validation_set_labels =  np.array(cifar_shuffled_labels[:5000])

    cifar_training_set_fourty_five_thousand =  np.array(cifar_shuffled_values[5000:])
    cifar_training_set_labels =  np.array(cifar_shuffled_labels[5000:])


    print("cifar validation shape: " + str(cifar_validation_set_five_thousand.shape))
    print("cifar validation labels shape: " + str(cifar_validation_set_labels.shape))

    print("cifar training shape: " + str(cifar_training_set_fourty_five_thousand.shape))
    print("cifar training labels shape: " + str(cifar_training_set_labels.shape))







shuffle_and_split_data()
