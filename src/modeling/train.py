from datasets import load_from_disk
from transformers import DataCollatorForSeq2Seq
from transformers import AutoTokenizer
from src.config import CHECKPOINT,BATCH_SIZE,MAX_LEN,D_K,D_MODEL,N_HEADS,N_LAYERS,EPOCHS
from torch.utils.data import DataLoader
from .architecture import Encoder,Decoder,Transformer
import torch
import torch.nn as nn
from datetime import datetime
import numpy as np

tokenized_dataset=load_from_disk('data/processed/tokenized_dataset')
tokenizer=AutoTokenizer.from_pretrained(CHECKPOINT)
tokenizer.add_special_tokens({'cls_token':'<s>'})
data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer)

train_loader=DataLoader(
    tokenized_dataset['train'],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=BATCH_SIZE
)

test_loader=DataLoader(
    tokenized_dataset['test'],
    batch_size=BATCH_SIZE,
    collate_fn=data_collator
)

encoder=Encoder(
    vocab_size=tokenizer.vocab_size+1,
    max_len=MAX_LEN,
    d_k=D_K,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    dropout_prob=0.1
)

decoder=Decoder(
    vocab_size=tokenizer.vocab_size+1,
    max_len=MAX_LEN,
    d_k=D_K,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    dropout_prob=0.1
)

model=Transformer(encoder,decoder)

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

encoder.to(device)
decoder.to(device)

criterion=nn.CrossEntropyLoss(ignore_index=-100)
optim=torch.optim.Adam(model.parameters())

def train(model,criterion,optimizer,train_loader,valid_loader,epochs):
  train_losses=np.zeros(epochs)
  test_losses=np.zeros(epochs)

  for it in range(epochs):
    model.train()
    t0=datetime.now()
    train_loss=[]
    for batch in train_loader:
      batch={k:v.to(device) for k,v in batch.items()}
      optimizer.zero_grad()

      enc_input=batch['input_ids']
      enc_mask=batch['attention_mask']
      targets=batch['labels']

      dec_input=targets.clone().detach()
      dec_input=torch.roll(dec_input,shifts=1,dims=1)
      dec_input[:,0]=65_001

      dec_input=dec_input.masked_fill(
          dec_input==-100, tokenizer.pad_token_id
      )

      dec_mask=torch.ones_like(dec_input)
      dec_mask=dec_mask.masked_fill(
          dec_input==tokenizer.pad_token_id, 0
      )

      outputs=model(enc_input,dec_input,enc_mask,dec_mask)
      loss=criterion(outputs.transpose(2,1),targets)

      loss.backward()
      optimizer.step()
      train_loss.append(loss.item())
    train_loss=np.mean(train_loss)
    model.eval()
    test_loss=[]
    for batch in valid_loader:
      batch={k:v.to(device) for k,v in batch.items()}
      enc_input=batch['input_ids']
      enc_mask=batch['attention_mask']
      targets=batch['labels']

      dec_input=targets.clone().detach()
      dec_input=torch.roll(dec_input,shifts=1,dims=1)
      dec_input[:,0]=65001

      dec_input=dec_input.masked_fill(
          dec_input==-100, tokenizer.pad_token_id
      )

      dec_mask=torch.ones_like(dec_input)
      dec_mask=dec_mask.masked_fill(
          dec_input==tokenizer.pad_token_id, 0
      )

      outputs=model(enc_input,dec_input,enc_mask,dec_mask)

      loss=criterion(outputs.transpose(2,1),targets)
      test_loss.append(loss.item())
    test_loss=np.mean(test_loss)

    train_losses[it]=train_loss
    test_losses[it]=test_loss

    dt=datetime.now()-t0
    print(f"Epoch: {it+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Duration: {dt}")
  return train_losses,test_losses

train_losses,test_losses=train(model,criterion,optim,train_loader,test_loader,EPOCHS)

torch.save(encoder.state_dict(),'models/encoder.pth')
torch.save(decoder.state_dict(),'models/decoder.pth')