import torch 
import torch.nn as nn 
import torch.nn.functionl as F

class modelForPredictions(nn.Module):
    def __init__(self, input_features, hidden_layer = 8, hidden_layer2 = 9, out_features = 3):
        super().__init__()  
        self.fc1 = nn.Linear(input_features, hidden_layer) #fully connected layer 1
        self.fc2 = nn.Linear(hidden_layer, hidden_layer2)
        self.out = nn.Linear(hidden_layer2, out_features)

    def forward(self, information_x):
        information_x = F.relu(self.fc1(x)) #Getting rid of < 0 options
        information_x = F.relu(self.fc2(x))
        information_x = self.out(x)
        return x

torch.manual_seed(41)
model = modelForPredictions()
