from keras.models import Sequential
from keras.layers import Activation
from keras.layers import LSTM
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import time
import psutil


def data_split_function(data_set):
    """
    data split percent: 0.8 training data, 0.2 testing data
    returns train_data and test_data
    """
    split_percent = 0.8
    split = int(split_percent * len(data_set))
    training_data = data_set[:split]
    testing_data = data_set[split:]
    return training_data, testing_data

def split_testing_data_to_input_output():
    """
    :return: the test data into input and output format
    """
    test_input_data = list()
    test_output_data = list()
    for seq_test in test_data:
        a = list()
        a.append(seq_test[0])
        a.append(seq_test[1])
        test_input_data.append(a)

    for seq_test in test_data:
        list2 = seq_test[2] + seq_test[3]
        test_output_data.append(list2)

    return test_input_data,test_output_data

def split_training_data_to_input_output():
    """
    :return: the train data into input and output format
    """
    train_input_data = list()
    train_output_data = list()
    for seq in train_data:
        list1 = list()
        list1.append(seq[0])
        list1.append(seq[1])
        train_input_data.append([list1])

    for seq in train_data:
        list1 = seq[2] + seq[3]
        train_output_data.append([list1])

    return train_input_data, train_output_data

def split_list(list_elements):
    """
    :param list_elements: elements of a single sequence
    :return: two parameters containing the sequence split in half
    """
    return list_elements[:4], list_elements[4:]

def convert_to_lstm_binary_format(data_set):
    """
    :param data_set: represents the entire data set that needs to be converted
    :return: the binary conversion of each element of the given sequence
    """
    final_list = list()
    for seq in data_set:
        list_one = list()
        # convert each element to a 4 bit binary number
        for elem in seq:
            list_one.append('{0:04b}'.format(elem))
        list_two = list()
        for elem in list_one:
            list_three = list()
            for character in elem:
                # convert char elements to int
                new_value = int(character)
                list_three.append(new_value)
            list_two.append(list_three)
        final_list.append(list_two)
    return final_list

def convert_binary_to_decimal(predicted_sequence):
    """
    :param predicted_sequence: binary input sequence
    :return: the conversion of the binary input sequence to the decimal format
    """
    binary_list = list()
    for element in predicted_sequence:
        for value in element:
            if value > 0.5:
                binary_list.append(1)
            else:
                binary_list.append(0)
    first_half, second_half = split_list(binary_list)
    res1 = int("".join(str(x) for x in first_half), 2)
    res2 = int("".join(str(x) for x in second_half), 2)
    return res1, res2

def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor

def get_memory_usage():
    """
    :return: the total physical memory used by the lstm model
    """
    memory = psutil.virtual_memory()
    print("memory ussage in bytes", get_size(memory.used))

def dataset_generator(x_train, y_train):
    """
    :param x_train: training input elements
    :param y_train: training output elements
    :return: the iterable tuple of input and output elements
    """
    i = 0
    while True:
        j = i % len(x_train)
        i += 1
        yield x_train[j], y_train[j]


start_time = time.time()
# load the data set into a data frame
df = pd.read_csv("C:\\Users\\GMB\\Desktop\\Licenta\\licenta backup\\data sets results\\1000 sequences\\dataset_LSTM.csv", names=["p1", "p2", "p3", "p4"])

values = df.values
train_data1, test_data1 = data_split_function(values)

# transform data to binary
train_data = convert_to_lstm_binary_format(train_data1)
test_data = convert_to_lstm_binary_format(test_data1)

train_input, train_output = split_training_data_to_input_output()
print('train data input is: {}'.format(train_input))
print('train data output is: {}'.format(train_output))

test_input,test_output = split_testing_data_to_input_output()
print('test data input is: {}'.format(test_input))
print('test data output is: {}'.format(test_output))

# generating the sequential model:
model = Sequential()

# adding the lstm layer
model.add(
    LSTM(units=8,
         batch_input_shape=(1, 2, 4),
         return_sequences=False
         ))

# adding the sigmoid activation function to the lstm layer
model.add(Activation('sigmoid'))

# compiling the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

# printing information about the model
model.summary()

# creating a generator for the train data, so the data can be iterated through the process

train_set_generator = dataset_generator(np.array(train_input), np.array(train_output))
history = model.fit_generator(train_set_generator,
                              steps_per_epoch=80,
                              epochs=85,
                              validation_data=(np.array(test_input), np.array(test_output)),
                              validation_steps=20)


prediction = model.predict(np.array([[[0, 0, 0, 1], [0, 1, 0, 0]]]))
pred1, pred2 = convert_binary_to_decimal(prediction)
print('for 1,4 the prediction is:{}, {}'.format(pred1,pred2))
print(prediction)
prediction1 = model.predict(np.array([[[0, 0, 0, 1], [0, 0, 1, 0]]]))
pred11, pred12 = convert_binary_to_decimal(prediction1)
print('for 1,2 the prediction is:{}, {}'.format(pred11,pred12))
print(prediction1)
prediction2 = model.predict(np.array([[[0, 0, 0, 1], [0, 0, 1, 1]]]))
pred21, pred22 = convert_binary_to_decimal(prediction2)
print('for 1,3 the prediction is:{}, {}'.format(pred21,pred22))
print(prediction2)
prediction3 = model.predict(np.array([[[0, 0, 0, 1], [0, 1, 1, 1]]]))
pred31, pred32 = convert_binary_to_decimal(prediction3)
print('for 1,7 the prediction is:{}, {}'.format(pred31,pred32))
print(prediction3)
prediction4 = model.predict(np.array([[[0, 0, 0, 1], [0, 0, 1, 0]]]))
pred41, pred42 = convert_binary_to_decimal(prediction4)
print('for 1,2 the prediction is:{}, {}'.format(pred41,pred42))
print(prediction4)
prediction5 = model.predict(np.array([[[0, 0, 0, 1], [0, 0, 1, 10]]]))
pred51, pred52 = convert_binary_to_decimal(prediction5)
print('for 1,4 the prediction is:{}, {}'.format(pred51,pred52))
print(prediction5)
prediction6 = model.predict(np.array([[[0, 0, 0, 1], [0, 1, 1, 1]]]))
pred61, pred62 = convert_binary_to_decimal(prediction6)
print('for 1,7 the prediction is:{}, {}'.format(pred61,pred62))
print(prediction6)
prediction7 = model.predict(np.array([[[0, 0, 0, 1], [0, 1, 0, 0]]]))
pred71, pred72 = convert_binary_to_decimal(prediction7)
print('for 1,4 the prediction is:{}, {}'.format(pred71,pred72))
print(prediction7)
prediction8 = model.predict(np.array([[[0, 0, 0, 1], [0, 0, 1, 0]]]))
pred81, pred82 = convert_binary_to_decimal(prediction8)
print('for 1,2 the prediction is:{}, {}'.format(pred81,pred82))
print(prediction8)
prediction9 = model.predict(np.array([[[0, 0, 0, 1], [0, 0, 1, 1]]]))
pred91, pred92 = convert_binary_to_decimal(prediction9)
print('for 1,3 the prediction is:{}, {}'.format(pred91,pred92))
print(prediction9)
resulted_list = list()
resulted_list.append(pred1)
resulted_list.append(pred2)
resulted_list.append(pred11)
resulted_list.append(pred12)
resulted_list.append(pred21)
resulted_list.append(pred22)
resulted_list.append(pred31)
resulted_list.append(pred32)
resulted_list.append(pred41)
resulted_list.append(pred42)
resulted_list.append(pred51)
resulted_list.append(pred52)
resulted_list.append(pred61)
resulted_list.append(pred62)
resulted_list.append(pred71)
resulted_list.append(pred72)
resulted_list.append(pred81)
resulted_list.append(pred82)
resulted_list.append(pred91)
resulted_list.append(pred92)

print("predicted list",resulted_list)
print("--- %s milliseconds ---" % ((time.time() - start_time)*1000))
get_memory_usage()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('noisy data - set model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()
