import os
os.chdir("..")
import torch
torch.manual_seed(0)
from guardrails.utils import NeuralNet, ToxicDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt



# --------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------

learning_rate = 0.001
momentum = 0.9
num_epochs = 100


# --------------------------------------------------------------------
# Dataset and DataLoader
# --------------------------------------------------------------------

# Training Dataset DataLoader        
train_labels_file = 'data/embeddings/train_labels.csv'
train_data_dir = 'data/embeddings/train/'

training_data = ToxicDataset(train_labels_file, train_data_dir)

train_loader = DataLoader(dataset=training_data,
                          batch_size=20,
                          shuffle=True,
                          num_workers=0)


# Validation Dataset DataLoader
val_labels_file = 'data/embeddings/val_labels.csv'
val_data_dir = 'data/embeddings/val/'
    
val_data = ToxicDataset(val_labels_file, val_data_dir)

val_loader = DataLoader(dataset=val_data,
                          batch_size=20,
                          shuffle=False,
                          num_workers=0)


# --------------------------------------------------------------------
# Train Feed-forward Neural Network Binary Classifier
# --------------------------------------------------------------------

net = NeuralNet(1024, 100, 25, 2)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()

# Optimizer updates model parameters
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

train_loss = []
val_loss = []

avrg_train_loss_per_epoch = []
avrg_val_loss_per_epoch = []


for epoch in range(num_epochs):

    # Training Loop
    temp_train_loss = []
    for i , (x, y) in enumerate(train_loader):    

        # Reset gradient
        optimizer.zero_grad()
        y_pred = net.forward(x)
        # calculate the losss
        loss = criterion(y_pred, y.long())
                
        train_loss.append(loss.item())
        temp_train_loss.append(loss.item())
        
        # gradient of the loss
        loss.backward()
        # Update parameters - adds the gradient to the weights 
        optimizer.step()
    # Below average train loss per epoch we use for plots
    avrg_train_loss = sum(temp_train_loss) / len(temp_train_loss)
    avrg_train_loss_per_epoch.append(avrg_train_loss)


    # Validation Loop
    temp_val_loss = []
    for i , (x, y) in enumerate(val_loader):

        y_val_pred = net.forward(x)

        # validation loss
        loss_val = criterion(y_val_pred, y.long())
        val_loss.append(loss_val.item())
        temp_val_loss.append(loss_val.item())
    # Use below for plotting val loss per epoch
    avrg_val_loss = sum(temp_val_loss) / len(temp_val_loss)        
    avrg_val_loss_per_epoch.append(avrg_val_loss)
    
    print("Epoch: {} | Avrg. Loss: {} | Avrg. Val Loss: {}".format(epoch, 
                                                                   round(avrg_train_loss,3), 
                                                                   round(avrg_val_loss,3)))

# Save the model    
torch.save(net.state_dict(), 'models/nn_1024_100_25_2')

# --------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------

epochs =[i for i in range(1, 101)]
plt.plot(epochs, avrg_train_loss_per_epoch, label='training loss')
plt.plot(epochs, avrg_val_loss_per_epoch, label='validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('')
plt.legend()
plt.savefig('models/nn_1024_100_25_loss.png')