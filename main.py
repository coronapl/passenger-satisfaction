"""
Pablo Valencia
A01700912
"""

import pandas as pd
from neuralNetwork import *
import matplotlib.pyplot as plt


global_accuracy = []
global_loss = []


def create_minibatches(batch_size, x, y):
    total_data = x.shape[0]
    return ((x[i:i + batch_size], y[i:i + batch_size]) for i in
            range(0, total_data, batch_size))


def clean_data(df):
    categorical_cols = [
        'Gender',
        'Customer Type',
        'Type of Travel',
        'Class'
    ]

    df.dropna(subset=['Arrival Delay in Minutes'], inplace=True)
    y = pd.DataFrame({'labels': df['satisfaction'].replace(
        to_replace=['neutral or dissatisfied', 'satisfied'], value=[0, 1])})

    df = pd.get_dummies(df, columns=categorical_cols)
    df = df.drop(df.columns[0], axis=1)
    x = df.drop(columns=['id', 'satisfaction', 'Gate location'])

    return x.to_numpy().astype(float), y.to_numpy()


def load_data(path):
    data = pd.read_csv(path)
    return clean_data(data)


def accuracy(x_data, y_data, batch_size):
    correct = 0
    total = 0
    for x, y in create_minibatches(batch_size, x_data, y_data):
        scores, Z2, Z1, A2, A1 = forward(x.T, parameters, [relu, relu])
        y_hat, _ = x_entropy(scores, y, len(x))
        correct += np.sum(np.argmax(y_hat, axis=0) == y.squeeze())
        total += y_hat.shape[1]
    return correct / total

x, y = load_data('data/train.csv')
# Shape x -> (100000, 27), y -> (100000, 1)
x_train, y_train = x[:100000], y[:100000]
# Shape x -> (3594, 27), y -> (3594, 1)
x_val, y_val = x[100000:], y[100000:]
# Shape x -> (25893, 27), y -> (25893, 1)
x_test, y_test = load_data('data/test.csv')

# Training hyperparameters
batch_size = 64
learning_rate = 0.001
epochs = 100
neurons = [26, 8, 8, 2]

parameters = init_parameters(neurons)
test_pretraining_accuracy = accuracy(x_test, y_test, batch_size)
global_accuracy.append(test_pretraining_accuracy)

scores1, Z2, Z1, A2, A1 = forward(x_train[:64].T, parameters, [relu, relu])
y_hat, cost = x_entropy(scores1, y_train[:64], batch_size)
global_loss.append(cost)
grads = backward(y_hat, x_train[:64], y_train[:64], Z2, A2, Z1, A1, parameters, batch_size)

# Training part
for epoch in range(epochs):
    for i, (x, y) in enumerate(create_minibatches(batch_size, x_train, y_train)):
        scores, Z2, Z1, A2, A1 = forward(x.T, parameters, [relu, relu])
        y_hat, cost = x_entropy(scores, y, len(x))
        grads = backward(y_hat, x, y, Z2, A2, Z1, A1, parameters, batch_size)

        # Update parameter after backtracking
        parameters['W1'] = parameters['W1'] - learning_rate * grads['W1']
        parameters['b1'] = parameters['b1'] - learning_rate * grads['b1']
        parameters['W2'] = parameters['W2'] - learning_rate * grads['W2']
        parameters['b2'] = parameters['b2'] - learning_rate * grads['b2']
        parameters['W3'] = parameters['W3'] - learning_rate * grads['W3']
        parameters['b3'] = parameters['b3'] - learning_rate * grads['b3']

    epoch_accuracy = accuracy(x_val, y_val, batch_size)
    global_accuracy.append(epoch_accuracy)
    global_loss.append(cost)
    print(f'Loss: {cost}, accuracy: {epoch_accuracy}')

# Final Accuracy after training
test_accuracy = accuracy(x_test, y_test, batch_size)
train_accuracy = accuracy(x_train, y_train, batch_size)

print('---- MODEL INFO ----')
print('--------------------')
print(f'Train Accuracy: {train_accuracy}')
print('Test data accuracy:')
print(f'Before training: {test_pretraining_accuracy}')
print(f'After training: {test_accuracy}')

epochs = range(len(global_accuracy))

plt.plot(epochs, global_accuracy)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Epochs')
plt.ylim(0, 1)
plt.show()

plt.plot(epochs, global_loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs. Epochs')
plt.show()

stop = False
while not stop:
        user_data = []
        print('----------------------')
        user_data.append(int(input('Age: ')))
        user_data.append(int(input('Flight Distance: ')))
        user_data.append(int(input('Inflight wifi service: ')))
        user_data.append(int(input('Departure/Arrival time convenient: ')))
        user_data.append(int(input('Ease of Online booking: ')))
        user_data.append(int(input('Food and drink: ')))
        user_data.append(int(input('Online boarding: ')))
        user_data.append(int(input('Seat comfort: ')))
        user_data.append(int(input('Inflight entertainment: ')))
        user_data.append(int(input('On-board service: ')))
        user_data.append(int(input('Leg room service: ')))
        user_data.append(int(input('Baggage handling: ')))
        user_data.append(int(input('Checkin service: ')))
        user_data.append(int(input('Inflight service: ')))
        user_data.append(int(input('Cleanliness: ')))
        user_data.append(int(input('Departure Delay in Minutes: ')))
        user_data.append(int(input('Arrival Delay in Minutes: ')))
        user_data.append(int(input('Gender_Female: ')))
        user_data.append(int(input('Gender_Male: ')))
        user_data.append(int(input('Customer Type_Loyal Customer: ')))
        user_data.append(int(input('Customer Type_disloyal Customer: ')))
        user_data.append(int(input('Type of Travel_Business travel: ')))
        user_data.append(int(input('Type of Travel_Personal Travel: ')))
        user_data.append(int(input('Class_Business: ')))
        user_data.append(int(input('Class_Eco: ')))
        user_data.append(int(input('Class_Eco Plus: ')))

        data = pd.DataFrame([user_data]).to_numpy()
        prediction, _, _, _, _ = forward(data.T, parameters, [relu, relu])
        print(f'Prediction: {np.argmax(prediction)}')

        stop_input = input('Do you want to make another prediction? yes/no\n')
        stop = True if stop_input == 'no' else False