# clap.py - Assicurati che TextEncoder usi l'argomento text_model
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel
from .audio import get_audio_encoder

# Rimuovi o commenta la variabile 'root' se esiste e non è usata correttamente
# ROOT_VARIABLE_EXISTS = False # Flag per sapere se l'abbiamo trovata
# try:
#     if 'root' in globals() or 'root' in locals(): # Controlla se 'root' è definita
#         print("Variabile 'root' trovata in clap.py, assicurati non venga usata per path hardcoded in TextEncoder.")
#         ROOT_VARIABLE_EXISTS = True
# except NameError:
#     pass # root non definita, va bene

class Projection(nn.Module):
    def __init__(self, d_in: int, d_out: int, p: float=0.5) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False)
        self.linear2 = nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        return embeds

class AudioEncoder(nn.Module):
    def __init__(self, audioenc_name:str, d_in: int, d_out: int, sample_rate: int, window_size: int,
            hop_size: int, mel_bins: int, fmin: int, fmax: int, classes_num: int) -> None:
        super().__init__()
        audio_encoder = get_audio_encoder(audioenc_name)
        self.base = audio_encoder(
            sample_rate, window_size,
            hop_size, mel_bins, fmin, fmax,
            classes_num, d_in)
        self.projection = Projection(d_in, d_out)

    def forward(self, x):
        out_dict = self.base(x)
        audio_features, audio_classification_output = out_dict['embedding'], out_dict['clipwise_output']
        projected_vec = self.projection(audio_features)
        return projected_vec, audio_classification_output

class TextEncoder(nn.Module): # QUESTA È LA CLASSE CRITICA
    # Il costruttore deve accettare 'text_model' e usarlo
    def __init__(self, d_out: int, text_model: str, transformer_embed_dim: int) -> None:
        super().__init__()
        print(f"TextEncoder (in clap.py) sta caricando il modello: '{text_model}' via AutoModel.from_pretrained")
        # Assicurati che QUALSIASI path hardcoded o uso di 'root' sia RIMOSSO da qui:
        self.base = AutoModel.from_pretrained(text_model) # DEVE USARE l'argomento text_model
        self.projection = Projection(transformer_embed_dim, d_out)

    def forward(self, x):
        out = self.base(**x)[0]
        out = out[:, 0, :]
        projected_vec = self.projection(out)
        return projected_vec

class CLAP(nn.Module): # Questa classe è usata se il YAML target è CLAP, non FrozenCLAPFLANEmbedder
    def __init__(self,
                audioenc_name: str, sample_rate: int, window_size: int, hop_size: int,
                mel_bins: int, fmin: int, fmax: int, classes_num: int, out_emb: int,
                text_model: str, transformer_embed_dim: int, d_proj: int):
        super().__init__()
        self.audio_encoder = AudioEncoder(
            audioenc_name, out_emb, d_proj,
            sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num)
        self.caption_encoder = TextEncoder(
            d_proj, text_model, transformer_embed_dim
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, audio, text):
        audio_embed, _ = self.audio_encoder(audio)
        caption_embed = self.caption_encoder(text)
        return caption_embed, audio_embed, self.logit_scale.exp()