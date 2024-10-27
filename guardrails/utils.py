import torch.nn.functional as F
import pickle
import ollama
import torch
import torchvision
import torch.utils.data
from torch.utils.data import Dataset
import itertools
import pandas as pd
import torch.nn as nn
import time
from random import randrange

class Utils(object):

    def __init__(self):
        pass


    def get_sentence_embedding(self, tokenizer:object, model:object, text:str) -> object:
        """
        Create sentence embeeding - for a single sentence
        Args: 
            - tokenizer
            - model: embedding model
            - text: sentence
        Return:
            - embedding: tensor object
        """

        # Tokenize the input texts
        batch_dict = tokenizer(text, max_length=8192, padding=True, truncation=True, return_tensors='pt')

        outputs = model(**batch_dict)
        embedding = outputs.last_hidden_state[:, 0]
        
        # normalize:
        embedding_normalized = F.normalize(embedding, p=2, dim=1)
        return embedding_normalized



    def create_embeddings(self, tokenizer:object, model:object, df:object, file_dir:str) -> None:
        """
        Create sentence embeedings for each comment in dataframe column "comment".
        Loops through the dataframe and embeds the comments.
        Each embedding is saved as a seperate pickle file that makes it easier to use it in pytorch
        Dataset and DataLoader modules.
        Args: 
            - tokenizer
            - model: embedding model
            - df: dataframe that contains comments (text) that we want to embed
            - file_dir: directory for saving embeddings
        Return:
            - None
        """

        for i in range(len(df)):
            # title - id of the sample
            title = df.iloc[[i]]['id'].values[0]
            txt = df.iloc[[i]]['comment_text'].values[0]
            
            #file_name = file_dir + title
            embedding = self.get_sentence_embedding(tokenizer=tokenizer, 
                                                    model=model, 
                                                    text=txt).data
            
            # Store data (serialize)
            #with open('{}.pickle'.format(file_name), 'wb') as handle:
            #    pickle.dump(embedding, handle, protocol=pickle.HIGHEST_PROTOCOL)

            self.save(data=embedding, dir=file_dir, file_name=title)

            

    def save(self, data:object, dir:str, file_name:str):
        """
        Utility function to save the data as pickle file 
        """

        with open('{}{}.pickle'.format(dir, file_name), 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
    def load(self, dir:str, file_name:str):
        """
        Utility function to load the data from pickle file 
        """
        
        with open('{}.pickle'.format(dir+file_name), 'rb') as handle:
                file = pickle.load(handle)
        return file
    

    def load_embedding_file(self, df_labels:object, dir:str, idx:int):

        file_name = df_labels.iloc[[idx]]['id'].values[0]
        label = df_labels.iloc[[idx]]['label'].values[0]
        embedding = self.load(dir=dir,
                              file_name = file_name)
        return embedding, label

    

class LLMClassifier(object):

    def __init__(self):
        pass


    def get_llm_response(self, model_name:str, input:str) -> str:
        """
        Generate LLM response for a given input. Here we use ollama for running LLama3.1 7B 
        locally.
        Args: 
            - model_name: 'llama3.1'
            - input: prompt
        Return:
            - output - response from LLM
        """

        response = ollama.chat(model=model_name, messages=[
                    {'role': 'user',
                    'content': '{}'.format(input)}])
        output = response['message']['content']
        return output
    

    def run_llm_llama_guard_classifier(self, df:object, 
                           df_labels:object,
                           prompt:str,
                           model_name:str):        
        y_true, y_pred, log = [], [], []
        for i in range(len(df_labels)):
            idx = df_labels.iloc[[i]]['id'].values[0]
            label = df_labels.iloc[[i]]['label'].values[0]
            message = df['comment_text'].loc[df['id'] == idx].values[0]
            message_prompt = prompt.format(USER_COMMENT=message)
            output = self.get_llm_response(model_name=model_name,
                                           input=message_prompt)
            y_pred.append(output)
            y_true.append(label)
            log_info = { "y_pred": output,
                "y_true": label,
                "message": message}
            log.append(log_info)
        return y_pred, y_true, log
        


    def run_llm_classifier(self, df:object, 
                           df_labels:object,
                           prompt:str,
                           model_name:str):
        """
        Generate LLM response for all comments in the dataframe column 'comment_text'. 
        Loops through all the rows in the collumn and creates classification using LLm + prompt
        See details on the prompt in prompts/prompt.py
        Args: 
            - df: orginal dataframe that contains all comments
            - df_labels: dataframe that contains only a subset data from df - balanced set of 1 and 0 labels
            - model_name: 'llama3.1'
            - prompt: prompt that is given to the LLM together with comment_text (message)
        Return:
            - y_pred - response from LLM with classifiction 1, 0 or 2 (if the LLM does not know the answer)
            - y_true - true label
            - log - the log of all outputs together with corresponding comment_text (message)
        """
        y_true, y_pred, log = [], [], []
        for i in range(len(df_labels)):
            idx = df_labels.iloc[[i]]['id'].values[0]
            label = df_labels.iloc[[i]]['label'].values[0]
            message = df['comment_text'].loc[df['id'] == idx].values[0]
            message_prompt = prompt.format(USER_COMMENT=message)

            output = self.get_llm_response(model_name=model_name,
                                           input=message_prompt)
            
            if not output.isnumeric():
                print("Failed: ", i, label, output)
                output = 2

            y_pred.append(int(output))
            y_true.append(label)
            log_info = { "y_pred": int(output),
                "y_true": label,
                "message": message}
            log.append(log_info)

        return y_pred, y_true, log
    


    def run_llm_gpt_classifier(self,
                               llm:object, 
                               df:object, 
                           df_labels:object,
                           prompt:str):
        
        y_true, y_pred, log = [], [], []
        for i in range(len(df_labels)):
            idx = df_labels.iloc[[i]]['id'].values[0]
            label = df_labels.iloc[[i]]['label'].values[0]
            message = df['comment_text'].loc[df['id'] == idx].values[0]
            message_prompt = prompt.format(USER_COMMENT=message)

            output = llm.invoke(message_prompt).content
            
            if not output.isnumeric():
                print("Failed: ", i, label, output)
                output = 2

            y_pred.append(int(output))
            y_true.append(label)
            log_info = { "y_pred": int(output),
                "y_true": label,
                "message": message}
            log.append(log_info)
            print(i, message)
            #time.sleep(randrange(10))

        return y_pred, y_true, log
        



class ToxicDataset(Dataset):
    """
    Customize torch Dataset object
    """

    def __init__(self, all_labels_file, x_files_dir):
        #load the labels file:
        self.labels = pd.read_csv(all_labels_file, index_col=0)
        
        # directory for all files
        self.x_files_dir = x_files_dir
        

    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        
        # Load data (deserialize)
        file_name = self.labels.iloc[[idx]]['id'].values[0]
        label = self.labels.iloc[[idx]]['label'].values[0]
        #print(file_name)
        
        with open('{}.pickle'.format(self.x_files_dir+file_name), 'rb') as handle:
            x = pickle.load(handle)[0]

        return x, torch.tensor(label) 
    


class NeuralNet(nn.Module):
    """
    Feed forward Neural Network architecture
    """
    def __init__(self, input_size, hidden1, hidden2, output_size):
        super(NeuralNet, self).__init__()
        
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.output_size = output_size
        
        self.i2h = nn.Linear(self.input_size, self.hidden1)
        self.h2h = nn.Linear(self.hidden1, self.hidden2)
        self.h2o = nn.Linear(self.hidden2, self.output_size)
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        x = F.relu(self.i2h(x))
        x = F.relu(self.dropout(self.h2h(x)))
        x = self.h2o(x)
        return x
    
