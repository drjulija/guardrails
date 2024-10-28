import os
os.chdir("..")
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from guardrails.utils import Utils
import pandas as pd

utils = Utils()


# --------------------------------------------------------------------
# # Load the tokenizer and embedding model
# --------------------------------------------------------------------

model_path = 'Alibaba-NLP/gte-large-en-v1.5'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

# --------------------------------------------------------------------
# Create training dataset
# --------------------------------------------------------------------

# 1. Load training data
df_train = pd.read_csv('data/balanced_train.csv')

# 2. Print dataset information
print("Train dataset info:")
print(df_train['label'].value_counts())
print(df_train.head(1))

# 3. Create embeddings and store in below directory
train_embed_dir = 'data/embeddings/train/'

utils.create_embeddings(tokenizer=tokenizer, 
                        model=model, 
                        df = df_train, 
                        file_dir=train_embed_dir)

# 4. Create labels dataset and store in below directory
df_train_labels = df_train[['id','label']]
df_train_labels.to_csv("data/embeddings/train_labels.csv")

del df_train, df_train_labels, train_embed_dir


# --------------------------------------------------------------------
# Create test dataset
# --------------------------------------------------------------------

# 1. Load test data
df_test = pd.read_csv('data/test.csv')  

# 2. Print dataset information
print("Test dataset info:")
print(df_test['label'].value_counts())
print(df_test.head(1))

# 3. Create balanced dataset - we want to create a balanced 
# test dataset with 3000 samples only
# We use random sampling to get below:
# - 1500 samples with label 0
# - 1500 samples with label 1
n = 1500
df_1 = df_test.loc[df_test['label'] == 1]
df_0 = df_test.loc[df_test['label'] == 0]
df_0 = df_0.sample(n=n, random_state=1)
df_1 = df_1.sample(n=n, random_state=1)
df_test_small = pd.concat([df_0, df_1]).sample(frac = 1).reset_index(drop=True)

# 4. Create embeddings and store in below directory
test_embed_dir = 'data/embeddings/test/'
utils.create_embeddings(tokenizer=tokenizer, 
                        model=model, 
                        df = df_test_small, 
                        file_dir=test_embed_dir)

# 5. Create labels dataset and store in below directory
df_test_small_labels = df_test_small[['id','label']]
df_test_small_labels.to_csv("data/embeddings/test_labels.csv")

del df_test, df_1, df_0, df_test_small, test_embed_dir, df_test_small_labels


# --------------------------------------------------------------------
# Create validation dataset
# --------------------------------------------------------------------

# 1. Load validation data
df_val = pd.read_csv('data/validation.csv')  

# 2. Print dataset information
print("Validation dataset info:")
print(df_val['label'].value_counts())
print(df_val.head(1))

# 3. Create balanced dataset - we want to create a balanced 
# validation dataset. We use random sampling to get below: 
# - 3291 samples with label 0
# - 3291 samples with label 1

df_1 = df_val.loc[df_val['label'] == 1]
df_0 = df_val.loc[df_val['label'] == 0]
df_0 = df_0.sample(n=len(df_1), random_state=1)

df_val_small = pd.concat([df_0, df_1]).sample(frac = 1).reset_index(drop=True)

# 4. Create embeddings and store in below directory
val_embed_dir = 'data/embeddings/val/'
utils.create_embeddings(tokenizer=tokenizer, 
                        model=model, 
                        df = df_val_small, 
                        file_dir=val_embed_dir)

# 5. Create labels dataset and store in below directory
df_val_small_labels = df_val_small[['id','label']]
df_val_small_labels.to_csv("data/embeddings/val_labels.csv")