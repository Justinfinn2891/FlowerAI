import torch 
import torch.nn as nn 
import torch.nn.functional as F
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class modelForPredictions(nn.Module):
    def __init__(self, input_features = 4, hidden_layer = 8, hidden_layer2 = 9, out_features = 3):
        super().__init__()  
        self.fc1 = nn.Linear(input_features, hidden_layer) #fully connected layer 1
        self.fc2 = nn.Linear(hidden_layer, hidden_layer2)
        self.out = nn.Linear(hidden_layer2, out_features)

    def forward(self, information_x):
        information_x = F.relu(self.fc1(information_x)) #Getting rid of < 0 options
        information_x = F.relu(self.fc2(information_x))
        information_x = self.out(information_x)
        return information_x

torch.manual_seed(51)
model = modelForPredictions()

datacapture = './dataset/iris.csv'
dataframe = pd.read_csv(datacapture)

#manipulating the dataset to produce numbers intstead of string
dataframe['variety'] = dataframe['variety'].replace('Setosa', 0.0)
dataframe['variety'] = dataframe['variety'].replace('Versicolor', 1.0)
dataframe['variety'] = dataframe['variety'].replace('Virginica', 2.0)


x_information = dataframe.drop('variety', axis = 1)
y_information = dataframe['variety']

x_information = x_information.values
y_information = y_information.values

X_train, X_test, y_train, y_test = train_test_split(x_information, y_information, test_size = .2, random_state = 41)

#turning them into tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

#Set the criterion 
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05) 

losses = []
epoch = 300

for i in range(epoch):
    y_pred = model.forward(X_train) #Get predicted results
    loss = criterion(y_pred, y_train) 
    losses.append(loss.detach().numpy()) #turning the tensor into a numpy array
    if i % 10 == 0:
        print(f'Epoch {i} and loss {loss}')
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(range(epoch), losses)
plt.ylabel("Loss/error")
plt.xlabel("Epoch")
plt.show()

with torch.no_grad(): 
    y_eval = model.forward(X_test)
    loss = criterion(y_eval, y_test)

correct = 0 
with torch.no_grad():
    for i, data in enumerate(X_test):
        y_val = model.forward(data)
        
        results = y_test[i]
        
        if y_test[i] == 0:
            results = "Setosa"
        if y_test[i] == 1:
            results = "Versicolor"
        if y_test[i] == 2:
            results = "Virginica"
        print(f'{i + 1}.) {str(y_val)} and {results}')
        if y_val.argmax().item() == y_test[i]:
            correct += 1

print(f'We have {correct} correct')
torch.save(model.state_dict(), 'flowers_AI')

new_model = modelForPredictions()
new_model.load_state_dict(torch.load('flowers_AI'))
plt.plot(range(epoch), losses)
plt.ylabel("Loss/error")
plt.xlabel("Epoch")
plt.show()
