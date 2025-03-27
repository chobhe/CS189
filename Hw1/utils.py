import random 
def shuffle(labels, training_data):
    zipped_data = list(zip(labels, training_data))
    random.shuffle(zipped_data)
    
    shuffled_labels = [i[0] for i in zipped_data]
    shuffled_values = [i[1] for i in zipped_data]
    return shuffled_labels, shuffled_values