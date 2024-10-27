import os
os.chdir("..")
import pandas as pd
from guardrails.utils import LLMClassifier, Utils
from prompts.prompt import task_prompt
import time

model_name = 'llama3.1'

model = LLMClassifier()
utils = Utils()

# --------------------------------------------------------------------
# Use LLM to classify toxic and non-toxic content on Test Data
# --------------------------------------------------------------------

# Load test data
df_test = pd.read_csv('data/test.csv')  
df_test_small_labels = pd.read_csv('data/embeddings/test_labels.csv')  

y_pred, y_true, log = model.run_llm_classifier(df=df_test, 
                           df_labels=df_test_small_labels,
                           prompt=task_prompt['prompt'],
                           model_name=model_name)

# --------------------------------------------------------------------
# Save results
# --------------------------------------------------------------------

utils.save(data=y_pred, dir='experiments/llm/' , file_name='y_pred')
utils.save(data=y_true, dir='experiments/llm/' , file_name='y_true')
utils.save(data=log, dir='experiments/llm/' , file_name='log')

