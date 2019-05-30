'''
Based on https://github.com/taoxugit/AttnGAN/blob/master/code/model.py
'''

import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from .gan_model import *

class AttnGAN(BaseModel):
    text_encoder_fn = RNN_ENCODER
    def __init__(self, vocab_size):

        ########## ENCODERS ##########
        if cfg.TRAIN.NET_E == '':
            print('Error: no pretrained text-image encoders')
            return
        # load and freeze image encoder
        self.image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
        img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        image_encoder.load_state_dict(torch.load(img_encoder_path,
                                        map_location=lambda storage, loc: storage))
        for p in self.image_encoder.parameters():
            p.requires_grad = False
        print('Loaded image encoder from:', img_encoder_path)
        self.image_encoder.eval()

        # load and freeze text encoder
        self.text_encoder = self.text_encoder_fn(vocab_size, nhidden=cfg.TEXT.EMBEDDING_DIM)
        self.text_encoder.load_state_dict(torch.load(cfg.TRAIN.NET_E,
                                            map_location=lambda storage, loc: storage))
        text_encoder.load_state_dict(state_dict)
        for p in text_encoder.parameters():
            p.requires_grad = False
        print('Load text encoder from:', cfg.TRAIN.NET_E)
        text_encoder.eval()

        ########## G and D ##########

class AttnBertGAN(AttnGAN):
    text_encoder_fn = RNN_BERT_ENCODER
