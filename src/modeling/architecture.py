import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataset
import numpy as np
import matplotlib.pyplot as plt
import math

class MultiHeadAttention(nn.Module):
  def __init__(self,d_k,d_model,n_heads,max_len,causal=False):
    super().__init__()

    self.d_k=d_k
    self.n_heads=n_heads

    self.key=nn.Linear(d_model,d_k*n_heads)
    self.query=nn.Linear(d_model,d_k*n_heads)
    self.value=nn.Linear(d_model,d_k*n_heads)

    self.fc=nn.Linear(d_k*n_heads,d_model)

    self.causal=causal
    if causal:
      cm=torch.tril(torch.ones(max_len,max_len))
      self.register_buffer('causal_mask',cm.view(1,1,max_len,max_len))

  def forward(self,q,k,v,pad_mask=None):
    q=self.query(q)  # N x T_output x (d_k*h)
    k=self.key(k)  # N x T_input x (d_k*h)
    v=self.value(v)  # N x T_input x (d_k*h)

    N,T_output,T_input=q.shape[0],q.shape[1],k.shape[1]

    q=q.view(N,T_output,self.n_heads,self.d_k).transpose(1,2) # NxT_outputxHxD_k -> NxHxT_outputxD_k
    k=k.view(N,T_input,self.n_heads,self.d_k).transpose(1,2) # NxT_inputxHxD_k -> NxHxT_inputxD_k
    v=v.view(N,T_input,self.n_heads,self.d_k).transpose(1,2) # NxT_inputxHxD_k -> NxHxT_inputxD_k

    attn_scores=q @ k.transpose(-2,-1) / math.sqrt(self.d_k) # NxHxT_outputxD_k @ NxHxD_kxT_input -> NxHxT_outputxT_input

    if pad_mask is not None:
      attn_scores=attn_scores.masked_fill(
          pad_mask[:,None,None,:]==0,float('-inf')
      )

    if self.causal:
      attn_scores=attn_scores.masked_fill(
          self.causal_mask[:,:,:T_output,:T_input]==0,float('-inf')
      )

    attn_scores=F.softmax(attn_scores,dim=-1)

    A=attn_scores @ v # NxHxT_outputxT_input @ NxHxT_inputxD_k -> NxHxT_outputxD_k

    A=A.transpose(1,2).contiguous().view(N,T_output,self.n_heads*self.d_k) # NxHxT_outputxD_k -> NxT_outputxHxD_k -> NxT_outputx(H*D_k)

    return self.fc(A) # NxT_outputx(H*D_k) -> NxT_outputxD_Model

class EncoderBlock(nn.Module):
  def __init__(self,d_model,d_k,n_heads,max_len,dropout_prob=0.1):
    super().__init__()

    self.ln1=nn.LayerNorm(d_model)
    self.ln2=nn.LayerNorm(d_model)
    self.mha=MultiHeadAttention(d_k,d_model,n_heads,max_len)
    self.ann=nn.Sequential(
        nn.Linear(d_model,4*d_model),
        nn.GELU(),
        nn.Linear(4*d_model,d_model),
        nn.Dropout(dropout_prob)
    )
    self.dropout=nn.Dropout(p=dropout_prob)

  def forward(self,x,pad_mask=None):
    x=self.ln1(x+self.mha(x,x,x,pad_mask))
    x=self.ln2(x+self.ann(x))
    x=self.dropout(x)
    return x

class DecoderBlock(nn.Module):
  def __init__(self,d_model,d_k,n_heads,max_len,dropout_prob=0.1):
    super().__init__()

    self.ln1=nn.LayerNorm(d_model)
    self.ln2=nn.LayerNorm(d_model)
    self.ln3=nn.LayerNorm(d_model)
    self.mha1=MultiHeadAttention(d_k,d_model,n_heads,max_len,causal=True)
    self.mha2=MultiHeadAttention(d_k,d_model,n_heads,max_len)
    self.ann=nn.Sequential(
        nn.Linear(d_model,4*d_model),
        nn.GELU(),
        nn.Linear(4*d_model,d_model),
        nn.Dropout(dropout_prob)
    )
    self.dropout=nn.Dropout(p=dropout_prob)

  def forward(self,enc_input,dec_input,enc_mask=None,dec_mask=None):
    x=self.ln1(dec_input+self.mha1(dec_input,dec_input,dec_input,dec_mask))
    x=self.ln2(x+self.mha2(x,enc_input,enc_input,enc_mask))
    x=self.ln3(x+self.ann(x))
    x=self.dropout(x)
    return x

class PositionalEncoding(nn.Module):
  def __init__(self,d_model,max_len=2048,dropout_prob=0.1):
    super().__init__()
    self.dropout=nn.Dropout(dropout_prob)

    position=torch.arange(max_len).unsqueeze(1)
    exp_term=torch.arange(0,d_model,2)
    div_term=torch.exp(exp_term*((-math.log(10000))/d_model))
    pe=torch.zeros(1,max_len,d_model)
    pe[0,:,0::2]=torch.sin(position*div_term)
    pe[0,:,1::2]=torch.cos(position*div_term)
    self.register_buffer('pe',pe)

  def forward(self,x):
    x=x+self.pe[:,:x.size(1),:]
    return self.dropout(x)

class Encoder(nn.Module):
  def __init__(self,vocab_size,max_len,d_k,d_model,n_heads,n_layers,dropout_prob):
    super().__init__()

    self.embedding=nn.Embedding(vocab_size,d_model)
    self.p_e=PositionalEncoding(d_model,max_len,dropout_prob)
    transformer_blocks=[
        EncoderBlock(d_model,d_k,n_heads,max_len,dropout_prob) for _ in range(n_layers)
    ]
    self.transformer_blocks=nn.Sequential(*transformer_blocks)
    self.ln1=nn.LayerNorm(d_model)

  def forward(self,x,pad_mask=None):
    x=self.embedding(x)
    x=self.p_e(x)
    for block in self.transformer_blocks:
      x=block(x,pad_mask)
    x=self.ln1(x)
    return x

class Decoder(nn.Module):
  def __init__(self,vocab_size,max_len,d_k,d_model,n_heads,n_layers,dropout_prob):
    super().__init__()

    self.embedding=nn.Embedding(vocab_size,d_model)
    self.p_e=PositionalEncoding(d_model,max_len,dropout_prob)
    transformer_blocks=[
        DecoderBlock(d_model,d_k,n_heads,max_len,dropout_prob) for _ in range(n_layers)
    ]
    self.transformer_blocks=nn.Sequential(*transformer_blocks)
    self.ln1=nn.LayerNorm(d_model)
    self.fc=nn.Linear(d_model,vocab_size)

  def forward(self,enc_output,dec_input,enc_mask=None,dec_mask=None):
    x=self.embedding(dec_input)
    x=self.p_e(x)
    for block in self.transformer_blocks:
      x=block(enc_output,x,enc_mask,dec_mask)
    x=self.ln1(x)
    x=self.fc(x)
    return x

class Transformer(nn.Module):
  def __init__(self,encoder,decoder):
    super().__init__()
    self.encoder=encoder
    self.decoder=decoder

  def forward(self,enc_input,dec_input,enc_mask,dec_mask):
    enc_output=self.encoder(enc_input,enc_mask)
    dec_output=self.decoder(enc_output,dec_input,enc_mask,dec_mask)
    return dec_output