import numpy as np
import scipy.special as scp
import os
import matplotlib.pyplot as plt

class NeuralNetwork:
    
    def __init__(self, inputnodes, hiddennodes, outputnodes, learingrate, training=False):
        self.inodes = inputnodes
        self.onodes = outputnodes
        self.hnodes = hiddennodes
        self.lr = learingrate
        self.tr = training
        self.W_ih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.W_ho = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        self.activation_function = lambda x: scp.expit(x)

    def train(self, inputs_list, targets_list):

        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        hidden_inputs = np.dot(self.W_ih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.W_ho, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.W_ho.T, output_errors)

        self.W_ho += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                      np.transpose(hidden_outputs))

        self.W_ih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                      np.transpose(inputs))
        return output_errors**2

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        hidden_inputs = np.dot(self.W_ih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.W_ho, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    def save(self):
        os.remove(r"C:/Users/Son YeongGwang/OneDrive/바탕 화면/weight_ih.csv")
        os.remove(r"C:/Users/Son YeongGwang/OneDrive/바탕 화면/weight_ho.csv")

        np.savetxt('C:/Users/Son YeongGwang/OneDrive/바탕 화면/weight_ih.csv', self.W_ih, fmt='%s', delimiter=',')
        np.savetxt('C:/Users/Son YeongGwang/OneDrive/바탕 화면/weight_ho.csv', self.W_ho, fmt='%s', delimiter=',')

    def load(self):
        w_ih = np.empty((0, self.inodes), float)
        w_ho = np.empty((0, self.hnodes), float)
        W_ih_list = open('C:/Users/Son YeongGwang/OneDrive/바탕 화면/weight_ih.csv', 'r')
        W_ho_list = open('C:/Users/Son YeongGwang/OneDrive/바탕 화면/weight_ho.csv', 'r')
        W_ih_data = W_ih_list.readlines()
        W_ho_data = W_ho_list.readlines()

        for record1 in W_ih_data:
            all_datas = record1.split(',')
            datas = np.asfarray(all_datas).reshape(1, self.inodes)
            w_ih = np.append(w_ih, datas, axis=0)

        for record2 in W_ho_data:
            all_datas = record2.split(',')
            datas = np.asfarray(all_datas).reshape(1, self.hnodes)
            w_ho = np.append(w_ho, datas, axis=0)

        self.W_ih = w_ih
        self.W_ho = w_ho

    def reTrain(self):
        if self.tr:
            return 1
        else:

            return 0


input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.1
losses = []
total_loss = []
n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate, training=True)

if n.reTrain():

    epoch = 10
    training_data_file = open("C:/Users/Son YeongGwang/OneDrive/바탕 화면/mnist_train.csv", "r")
    training_data_list = training_data_file.readlines()
    training_data_list_2d = np.array(training_data_list, ndmin=2)
    training_data_file.close()
    i = 0
    for e in range(epoch):
        i = i + 1

        for j, record in enumerate(training_data_list):
            all_values = record.split(',')
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            targets = np.zeros(output_nodes) + 0.01
            targets[int(record[0])] = 0.99
            if j == 0 or j % 1000 == 0:
                print("[INFO] Training Model...Total {}/{} in {:.2f}%".format(i, epoch, (j / len(training_data_list_2d[0]) * 100)))

            loss = n.train(inputs, targets)
            losses.append(loss)

        total_loss.append(np.sum(losses) / len(losses))
    n.save()

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, epoch), total_loss)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

else:
    n.load()

testing_data_file = open("C:/Users/Son YeongGwang/OneDrive/바탕 화면/mnist_test.csv", "r")
testing_data_list = testing_data_file.readlines()
testing_data_file.close()

scorecard = []

for record in testing_data_list:

    test_values = record.split(',')
    test_input = np.asfarray(test_values[1:])

    # image_array = np.asfarray(test_values[1:]).reshape(28, 28)
    # print("Input Number  :", test_values[0])
    # print("Prediction    :", np.argmax(n.query(test_input)), "\n")

    if int(test_values[0]) == np.argmax(n.query(test_input)):
        scorecard.append(1)
    else:
        scorecard.append(0)

# print(scorecard)
print("accuracy: "+str(np.mean(scorecard)*100)+"%\n")

## Testing from personal data
# image_array = np.asfarray(Image.open("C:/Users/Son YeongGwang/OneDrive/바탕 화면/test.png").convert('L'))
# image_data = 255.0 - image_array.reshape(784)
# image_data = (image_data / 255 * 0.99) + 0.01
#
# print("NN says:", np.argmax(n.query(image_data)))
