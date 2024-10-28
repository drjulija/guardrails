import os
os.chdir("..")
import pandas as pd
from guardrails.utils import Utils, NeuralNet
import time
import torch
import pickle

utils = Utils()


# --------------------------------------------------------------------
# Use NN model to classify toxic and non-toxic content on Test Data
# --------------------------------------------------------------------

# Load test data labels and embeddings
df_test_small_labels = pd.read_csv('data/embeddings/test_labels.csv', index_col=0)  
test_data_dir = 'data/embeddings/test/'


# Load the trained NN model
model = NeuralNet(1024, 100, 25, 2)
model.load_state_dict(torch.load('models/nn_1024_100_25_2', weights_only=True))
model.eval()


y_pred, y_true = [], []

for idx in range(len(df_test_small_labels)):
    embedding, label = utils.load_embedding_file(df_labels=df_test_small_labels, 
                              dir=test_data_dir, 
                              idx=idx)
    with torch.no_grad():
        y_test_pred = model(embedding)
    y_pred.append(torch.argmax(y_test_pred).item())
    y_true.append(label)


# --------------------------------------------------------------------
# Save results
# --------------------------------------------------------------------

utils.save(data=y_pred, dir='experiments/nn/' , file_name='y_pred')
utils.save(data=y_true, dir='experiments/nn/' , file_name='y_true')
