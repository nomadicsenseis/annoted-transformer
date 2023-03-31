#Imports for Harvard PyTorch implementation
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context="talk")
%matplotlib inline
import os
from os.path import exists
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy
import time
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import altair as alt
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torchtext.datasets as datasets
import spacy
import GPUtil
import warnings
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

##Pytorch
class CapaNorm(nn.Module):
  """
  Se construye un modulo capa de normalización.
  """
  def __init__(self, caracteristicas,eps=1e-6):
    super(CapaNorm, self).__init__()
    self.a_2=nn.Parameter(torch.ones(caracteristicas))
    self.b_2=nn.Parameter(torch.zeros(caracteristicas))
    self.eps=eps

  def forward(self,x):
    mean=x.mean(-1,keepdim=True)
    std=x.std(-1,keepdim=True)
    return self.a_2*(x-mean)/(std+self.eps)+self.b_2

class ConexionSubCapa(nn.Module):
  """
  Una conexión residual seguida de una capa de normalización. 
  """

  def __init__(self, tamaño, dropout):
    super(ConexionSubCapa,self).__init__()
    norm=CapaNorm(tamaño)
    self.dropout=nn.Dropout(dropout)

  def forward(self, x, subcapa):
    "Aplica conexion residual a cualquier sub capa con el mismo tamaño."
    return x + self.dropout(subcapa(self.norm(x)))

def MaskUnidireccional(tamaño):
  attn_shape=(1,tamaño,tamaño)
  mask=np.triu(np.ones(attn_shape),k=1).astype('uint8')
  return torch.from_numpy(mask)==0

def attention(query, key, value, mask=None, dropout=None):
  d_k=query.size(-1)
  scores=torch.matmul(query,key.transpose(-2,-1)) / math.sqrt(d_k)
  if mask is not None:
    scores=scores.masked_fill(mask==0,-1e9)
    p_attn=F.softmax(scores,dim=-1)
  if dropout is not None:
    p_attn=dropout(p_attn)
  return torch.matmul(p_attn,value),p_attn

class MultiHeadedattention(nn.Module):
  def __init__(self,h,d_model,dropout=0.1):
    super(MultiHeadedattention,self).__init__()
    assert d_model % h ==0 
    self.d_k=d_model//h
    self.h=h
    self.linears=clones(nn.Linear(d_model,d_model),4)
    self.attn=None
    self.dropout=nn.Dropout(p=dropout)

  def forward(self, query, key, value, mask=None):
    if mask is not None:
      mask=mask.unsqueeze(1)
    nbatches=query.size(0)
  # 1) hacer todas las proyecciones lineales en lotes desde d_model => h x d_k
    query,key,value= [l(x).view(nbatches,-1,self.h, self.d_k).transpose(1,2) for l,x in zip(self.linears,(query,key,value))]
  # 2) Aplicar el mecanismod e atención en todos los vectores proyectdos por lotes.
    x, self.attn=attention(query,key,value,mask=mask,dropout=self.dropout)
  # 3) Concatenar utilizando un view y aplicar la capa lineal final.
    x=x.transpose(1,2).contiguous().view(nbatches,-1,self.h*self.d_k)

    return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
  def __init__(self,d_model,d_ff,dropout=0.1):
    super(PositionwiseFeedForward,self).__init__()
    self.w_1=nn.Linear(d_model,d_ff)
    self.w_2=nn.Linear(d_ff,d_model)
    self.dropout=nn.Dropout(dropout)

  def forward(self,x):
    return self.w_2(self.dropout(F.relu(self.w_1(x))))

def clones(module,N):
  """
  Produce N capas idénticas.
  """
  return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class CapaEncoder(nn.Module):
  def __init__(self,tamaño,self_attn,PFF,dropout):
    super(CapaEncoder,self).__init__()
    self.self_attn=self_attn
    self.PFF=PFF
    self.subcapa=clones(ConexionSubCapa(tamaño,dropout),2)
    self.tamaño=tamaño

  def forward(self, x,mask):
    x=self.subcapa[0](x,lambda x: self.self_attn(x,x,x,mask))
    return self.subcapa[1](x, self.PFF)

class Encoder(nn.Module):
  """
  El núcleo del encoder es un stack de N capas
  """
  def __init__(self,capa,N):
    super(Encoder, self).__init__()
    self.capas = clones(capa,N)
    self.norm= CapaNorm(capa.size)

  def forward(self,x,mask):
    """
    Pasa el input (y máscara) a traves de cada capa en respuesta.
    """
    for capa in self.capas:
      x=capa(x,mask)
    return self.norm(x)

class CapaDecoder(nn.Module):

  def __init__(self, tamaño, self_attn, cross_attn, PFF, dropout):
    super(CapaDecoder,self).__init__()
    self.tamaño=tamaño
    self.self_attn=self_attn
    self.cross_attn=cross_attn
    self.PFF=PFF
    self.subcapa=clones(ConexionSubCapa(tamaño,dropout),3)

  def forward(self,x,memoria,ori_mask,obj_mask):
    m=memoria
    x=self.subcapa[0](x, lambda x: self.self_atten(x,x,x,obj_mask))
    x=self.subcapa[1](x, lambda x: self.ori_mask(x,m,m, ori_mask))
    return self.subcapa[2](x,self.PFF)
class Decoder(nn.Module):
  """
  Decoder genérico con N capas y máscara.
  """
  def __init__(self, capa,N):
    super(Decoder,self).__init__()
    self.capas=clones(capa,N)
    self.norm=CapaNorm(capa.size)

  def forward(self,x, memoria, ori_mask, obj_mask):
    for capa in self.capas:
      x=capa(x,memoria,ori_mask,obj_mask)
    return self.norm(x)

class EncoderDecoder(nn.Module):
  """
  Arquitectura estándar Codificador-Decodificador. 
  """

  def __init__(self, encoder, decoder, ori_embed, obj_embed, generator):
    super(EncoderDecoder,self).__init__()
    self.encoder=encoder
    self.decoder=decoder
    self.ori_embed=ori_embed
    self.obj_embed=obj_embed
    self.generator=generator

  def forward(self,ori,obj,ori_mask,obj_mask):
    "Ingerir y procesar la seqcuencia de origen y la objetivo"
    return self.decode(self.encode(ori,ori_mask),ori_mask,obj,obj_mask)

  def encode(self, ori, ori_mask):
    return self.encoder(self, ori, ori_mask)

  def decode(self,memory, ori_mask, obj, obj_mask):
    return self.decoder(self.obj_embed(obj), memory, ori_mask, obj_mask)

class Embeddings(nn.Module):

  def __init__(self, d_model, vocab):
    super(Embeddings,self).__init__()
    self.lut=nn.Embedding(vocab,d_model)
    self.d_model=d_model

  def forward(self,x):
    return self.lut(x)*math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):

  def __init__(self,d_model,dropout,max_len=5000):
    super(PositionalEncoding,self).__init__()
    self.dropout=nn.Dropout(p=dropout)

    #Calcula los positional encodings una vez en el espacio logarítmico
    pe=torch.zeros(max_len, d_model)
    position=torch.arange(0,max_len).unsqueeze(1)
    div_term=torch.exp(torch.arange(0,d_model,2)*
                       -(math.log(10000.0)/d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    self.register_buffer('pe', pe)
        
  def forward(self, x):
    x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
    return self.dropout(x)

class Unembedding(nn.Module):
  """
  se define el paso Feed Forward
  """
  def __init__(self, d_model, vocab):
    super(Unembedding, self).__init__()
    self.proj=nn.Linear(d_model,vocab)

  def forward(self,x):
    return F.log_softmax(self.proj(x),dim=-1)

def make_model(
    ori_vocab, obj_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedattention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(CapaEncoder(d_model, c(attn), c(ff), dropout), N),
        Decoder(CapaDecoder(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, ori_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, obj_vocab), c(position)),
        Unembedding(d_model, obj_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

