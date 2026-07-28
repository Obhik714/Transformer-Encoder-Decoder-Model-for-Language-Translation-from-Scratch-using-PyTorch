import pandas as pd
from datasets import load_dataset
from .config import MAX_INPUT_LEN, MAX_OUTPUT_LEN, CHECKPOINT
from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained(CHECKPOINT)

def preprocess_function(batch):
  model_inputs=tokenizer(batch['en'],max_length=MAX_INPUT_LEN,truncation=True)
  targets=tokenizer(batch['es'],max_length=MAX_OUTPUT_LEN,truncation=True)
  model_inputs['labels']=targets['input_ids']
  return model_inputs

df=pd.read_csv('data/external/spa.txt',sep='\t', header=None)
df=df.iloc[:30000]
df.columns=['en','es']
df.to_csv('data/interim/spa.csv',index=None)
raw_dataset=load_dataset('csv',data_files='data/interim/spa.csv')
split=raw_dataset['train'].train_test_split(test_size=0.3,seed=42)
tokenized_ds=split.map(preprocess_function,batched=True,remove_columns=split['train'].column_names)
tokenized_ds.save_to_disk("data/processed/tokenized_dataset")
