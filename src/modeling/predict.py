from .architecture import Encoder, Decoder
from src.config import CHECKPOINT,MAX_LEN,D_K,D_MODEL,N_HEADS,N_LAYERS
from transformers import AutoTokenizer
import torch

tokenizer=AutoTokenizer.from_pretrained(CHECKPOINT)
tokenizer.add_special_tokens({'cls_token':'<s>'})

encoder=Encoder(
    vocab_size=tokenizer.vocab_size+1,
    max_len=MAX_LEN,
    d_k=D_K,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    N_layers=N_LAYERS,
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

encoder.load_state_dict(torch.load('models/encoder.pth'))
decoder.load_state_dict(torch.load('models/decoder.pth'))

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

encoder.to(device)
decoder.to(device)

def translate(input_sentence):
  enc_input=tokenizer(input_sentence,return_tensors='pt').to(device)
  enc_output=encoder(enc_input['input_ids'],enc_input['attention_mask'])

  dec_input_str='<s>'

  dec_input=tokenizer(text_target=dec_input_str,return_tensors='pt').to(device)
  dec_input_ids=dec_input['input_ids'][:,:-1]
  dec_attn_mask=dec_input['attention_mask'][:,:-1]

  for _ in range(32):
    dec_output=decoder(
        enc_output,
        dec_input_ids,
        enc_input['attention_mask'],
        dec_attn_mask
    )
    prediction_id=torch.argmax(dec_output[:,-1,:],axis=-1)
    dec_input_ids=torch.hstack((dec_input_ids,prediction_id.view(1,1)))
    dec_attn_mask=torch.ones_like(dec_input_ids)

    if prediction_id==0:
      break
  return tokenizer.decode(dec_input_ids[0,1:-1])