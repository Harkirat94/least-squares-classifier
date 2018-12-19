import numpy as np
import csv
import sklearn.metrics as skm
import sys

class_label_to_vector_rep =	{
  "Iris-setosa": np.asarray([1,0,0]),
  "Iris-versicolor": np.asarray([0,1,0]),
  "Iris-virginica": np.asarray([0,0,1])
}

# reading data
data = []
with open('iris.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        modified_row = []
        for i, item in enumerate(row):
            if i!=4:
                modified_row.append(float(item))
            else:
                modified_row.append(item)
        data.append(modified_row)


setosa = data[:50]
versicolor = data[50:100]
virginica = data[100:150]


def divide_training_and_test(labeled_data, training_percentage):
    population_size = len(labeled_data)
    training_count = int((training_percentage/100.0)*population_size)
    test_count = population_size - training_count
    training_data = labeled_data[:training_count]
    test_data = labeled_data[training_count:]
    return training_data, test_data


def prepare_data(data_set):
    modified_data_set = []
    for xi in data_set:
        yi = class_label_to_vector_rep[xi[4]]
        xi[4] = 1
        modified_xi = np.asarray(xi)
        modified_data_set.append((modified_xi, yi))
    return modified_data_set


def class_vet_to_number(data):
    actual_classes = []
    for data_point in data:
        actual_classes.append(np.argmax(data_point))
    return actual_classes

def calculate_weight_matrix(train_set, lam):
    first_part = np.zeros((5, 5))
    for train_data_point in train_set:
        xi = train_data_point[0]
        xi = xi[:, np.newaxis]
        step_mul = np.matmul(xi, xi.transpose())
        first_part = np.add(first_part, step_mul)
    first_part += lam
    inverse_first_part = np.linalg.inv(first_part)
    second_part = np.zeros((5, 3))
    for train_data_point in train_set:
        xi = train_data_point[0]
        yi = train_data_point[1]
        xi = xi[:, np.newaxis]
        yi = yi[:, np.newaxis]
        step_mul = np.matmul(xi, yi.transpose())
        second_part = np.add(second_part, step_mul)
    weight_matrix = np.matmul(inverse_first_part, second_part)
    return weight_matrix


def remove_class_from_test_data(test_set):
    test_actual_class_labels = []
    class_removed_test_set = []
    for test_data_point in test_set:
        class_removed_test_set.append(test_data_point[0])
        test_actual_class_labels.append(test_data_point[1])
    return class_removed_test_set, test_actual_class_labels


def start_testing(weight_matrix, class_removed_test_set):
    divided_weight_matrix = np.hsplit(weight_matrix, 3)
    predicted_classes = []
    for xi in class_removed_test_set:
        xi = xi[:, np.newaxis]
        max_value = float("-inf")
        max_index = -1
        for i, wk in enumerate(divided_weight_matrix):
            if max_value < np.matmul(wk.transpose(), xi):
                max_index = i
                max_value = np.matmul(wk.transpose(), xi)
        predicted_classes.append(max_index)
        #print "max_index:",max_index, "and max_value:", max_value
    return predicted_classes


def create_confusion_matrix(actual, predicted):
    confusion_matrix = np.zeros((3, 3), dtype=int)
    for idx_actual, idx_predicted in zip(actual, predicted):
        confusion_matrix[idx_actual][idx_predicted] += 1
    return confusion_matrix


def classification(training_percent, lam):
    s_training, s_test = divide_training_and_test(setosa, training_percent)
    ve_training, ve_test = divide_training_and_test(versicolor, training_percent)
    vi_training, vi_test = divide_training_and_test(virginica, training_percent)

    training_set = s_training + ve_training + vi_training
    test_set = s_test + ve_test + vi_test

    modified_training_set = prepare_data(training_set)
    modified_test_set = prepare_data(test_set)
    class_removed_test_set, test_actual_class_vector = remove_class_from_test_data(modified_test_set)
    actual_classes = class_vet_to_number(test_actual_class_vector)
    weight_matrix = calculate_weight_matrix(modified_training_set, lam)

    predicted_classes = start_testing(weight_matrix, class_removed_test_set)
    # print predicted_classes
    confusion_matrix = create_confusion_matrix(actual_classes, predicted_classes)
    print "Confusion Matrix for training percentage", training_percent,\
        "and test percent", 100-training_percent, "with lambda", lam, "is:"
    print confusion_matrix

    class_1_accuracy = skm.accuracy_score(actual_classes[:len(s_test)],
                                          predicted_classes[:len(s_test)])
    print "Class-1 misclassification error is: ", 1-class_1_accuracy

    class_2_accuracy = skm.accuracy_score(actual_classes[len(s_test):2*len(s_test)],
                                          predicted_classes[len(s_test):2*len(s_test)])
    print "Class-2 misclassification error is: ", 1-class_2_accuracy

    class_3_accuracy = skm.accuracy_score(actual_classes[2*len(s_test):3*len(s_test)],
                                          predicted_classes[2*len(s_test):3*len(s_test)])
    print "Class-3 misclassification error is: ", 1-class_3_accuracy

    total_accuracy = skm.accuracy_score(actual_classes, predicted_classes)
    print "Combined misclassification error is:", 1-total_accuracy


user_training_ratio = int(sys.argv[1])
user_lambda = int(sys.argv[2])
if not 0 < user_training_ratio < 100:
    print "Enter the training ratio between 0 and 100"
    sys.exit(0)

classification(user_training_ratio, user_lambda)
