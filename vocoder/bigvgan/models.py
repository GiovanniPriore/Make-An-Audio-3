# Copyright (c) 2022 NVIDIA CORPORATION. 
#   Licensed under the MIT license.

# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.


import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
import numpy as np
from .activations import Snake,SnakeBeta
from .alias_free_torch import *
import os
from omegaconf import OmegaConf
import json # Per caricare config.json
import yaml # Per il fallback a args.yml

LRELU_SLOPE = 0.1

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

class AMPBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5), activation=None):
        super(AMPBlock1, self).__init__()
        self.h = h

        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

        self.num_layers = len(self.convs1) + len(self.convs2) # total number of conv layers

        if activation == 'snake': # periodic nonlinearity with snake function and anti-aliasing
            self.activations = nn.ModuleList([
                Activation1d(
                    activation=Snake(channels, alpha_logscale=h.snake_logscale))
                for _ in range(self.num_layers)
            ])
        elif activation == 'snakebeta': # periodic nonlinearity with snakebeta function and anti-aliasing
            self.activations = nn.ModuleList([
                Activation1d(
                    activation=SnakeBeta(channels, alpha_logscale=h.snake_logscale))
                 for _ in range(self.num_layers)
            ])
        else:
            raise NotImplementedError("activation incorrectly specified. check the config file and look for 'activation'.")

    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x

        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class AMPBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3), activation=None):
        super(AMPBlock2, self).__init__()
        self.h = h

        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

        self.num_layers = len(self.convs) # total number of conv layers

        if activation == 'snake': # periodic nonlinearity with snake function and anti-aliasing
            self.activations = nn.ModuleList([
                Activation1d(
                    activation=Snake(channels, alpha_logscale=h.snake_logscale))
                for _ in range(self.num_layers)
            ])
        elif activation == 'snakebeta': # periodic nonlinearity with snakebeta function and anti-aliasing
            self.activations = nn.ModuleList([
                Activation1d(
                    activation=SnakeBeta(channels, alpha_logscale=h.snake_logscale))
                 for _ in range(self.num_layers)
            ])
        else:
            raise NotImplementedError("activation incorrectly specified. check the config file and look for 'activation'.")

    def forward(self, x):
        for c, a in zip (self.convs, self.activations):
            xt = a(x)
            xt = c(xt)
            x = xt + x

        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class BigVGAN(torch.nn.Module):
    # this is our main BigVGAN model. Applies anti-aliased periodic activation for resblocks.
    def __init__(self, h):
        super(BigVGAN, self).__init__()
        self.h = h

        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        # pre conv
        self.conv_pre = weight_norm(Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3))

        # define which AMPBlock to use. BigVGAN uses AMPBlock1 as default
        resblock = AMPBlock1 if h.resblock == '1' else AMPBlock2

        # transposed conv-based upsamplers. does not apply anti-aliasing
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(nn.ModuleList([
                weight_norm(ConvTranspose1d(h.upsample_initial_channel // (2 ** i),
                                            h.upsample_initial_channel // (2 ** (i + 1)),
                                            k, u, padding=(k - u) // 2))
            ]))

        # residual blocks using anti-aliased multi-periodicity composition modules (AMP)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d, activation=h.activation))

        # post conv
        if h.activation == "snake": # periodic nonlinearity with snake function and anti-aliasing
            activation_post = Snake(ch, alpha_logscale=h.snake_logscale)
            self.activation_post = Activation1d(activation=activation_post)
        elif h.activation == "snakebeta": # periodic nonlinearity with snakebeta function and anti-aliasing
            activation_post = SnakeBeta(ch, alpha_logscale=h.snake_logscale)
            self.activation_post = Activation1d(activation=activation_post)
        else:
            raise NotImplementedError("activation incorrectly specified. check the config file and look for 'activation'.")

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))

        # weight initialization
        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        # pre conv
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            # upsampling
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)
            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # post conv
        x = self.activation_post(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            for l_i in l:
                remove_weight_norm(l_i)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class DiscriminatorP(torch.nn.Module):
    def __init__(self, h, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.d_mult = h.discriminator_channel_mult
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, int(32*self.d_mult), (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(int(32*self.d_mult), int(128*self.d_mult), (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(int(128*self.d_mult), int(512*self.d_mult), (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(int(512*self.d_mult), int(1024*self.d_mult), (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(int(1024*self.d_mult), int(1024*self.d_mult), (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(int(1024*self.d_mult), 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, h):
        super(MultiPeriodDiscriminator, self).__init__()
        self.mpd_reshapes = h.mpd_reshapes
        print("mpd_reshapes: {}".format(self.mpd_reshapes))
        discriminators = [DiscriminatorP(h, rs, use_spectral_norm=h.use_spectral_norm) for rs in self.mpd_reshapes]
        self.discriminators = nn.ModuleList(discriminators)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorR(nn.Module):
    def __init__(self, cfg, resolution):
        super().__init__()

        self.resolution = resolution
        assert len(self.resolution) == 3, \
            "MRD layer requires list with len=3, got {}".format(self.resolution)
        self.lrelu_slope = LRELU_SLOPE

        norm_f = weight_norm if cfg.use_spectral_norm == False else spectral_norm
        if hasattr(cfg, "mrd_use_spectral_norm"):
            print("INFO: overriding MRD use_spectral_norm as {}".format(cfg.mrd_use_spectral_norm))
            norm_f = weight_norm if cfg.mrd_use_spectral_norm == False else spectral_norm
        self.d_mult = cfg.discriminator_channel_mult
        if hasattr(cfg, "mrd_channel_mult"):
            print("INFO: overriding mrd channel multiplier as {}".format(cfg.mrd_channel_mult))
            self.d_mult = cfg.mrd_channel_mult

        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, int(32*self.d_mult), (3, 9), padding=(1, 4))),
            norm_f(nn.Conv2d(int(32*self.d_mult), int(32*self.d_mult), (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(int(32*self.d_mult), int(32*self.d_mult), (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(int(32*self.d_mult), int(32*self.d_mult), (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(int(32*self.d_mult), int(32*self.d_mult), (3, 3), padding=(1, 1))),
        ])
        self.conv_post = norm_f(nn.Conv2d(int(32 * self.d_mult), 1, (3, 3), padding=(1, 1)))

    def forward(self, x):
        fmap = []

        x = self.spectrogram(x)
        x = x.unsqueeze(1)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.lrelu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

    def spectrogram(self, x):
        n_fft, hop_length, win_length = self.resolution
        x = F.pad(x, (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)), mode='reflect')
        x = x.squeeze(1)
        x = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=False, return_complex=True)
        x = torch.view_as_real(x)  # [B, F, TT, 2]
        mag = torch.norm(x, p=2, dim =-1) #[B, F, TT]

        return mag


class MultiResolutionDiscriminator(nn.Module):
    def __init__(self, cfg, debug=False):
        super().__init__()
        self.resolutions = cfg.resolutions
        assert len(self.resolutions) == 3,\
            "MRD requires list of list with len=3, each element having a list with len=3. got {}".\
                format(self.resolutions)
        self.discriminators = nn.ModuleList(
            [DiscriminatorR(cfg, resolution) for resolution in self.resolutions]
        )

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(x=y)
            y_d_g, fmap_g = d(x=y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss*2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


class VocoderBigVGAN(nn.Module):  # Falla ereditare da nn.Module per coerenza
    def __init__(self, ckpt_vocoder_dir, device='cuda'):  # Rinominato per chiarezza
        super(VocoderBigVGAN, self).__init__()  # Chiama l'init della superclasse
        self.device = torch.device(device)  # Assicura che device sia un oggetto torch.device

        print(f"VocoderBigVGAN: Inizializzazione dalla directory: {ckpt_vocoder_dir}")

        # --- Caricamento Configurazione ---
        config_path_json = os.path.join(ckpt_vocoder_dir, "config.json")
        config_path_yml = os.path.join(ckpt_vocoder_dir, "args.yml")

        h = None  # Iperparametri
        if os.path.exists(config_path_json):
            print(f"  Trovato config.json, lo carico...")
            with open(config_path_json, "r") as f:
                data = json.load(f)

            # Converti il dizionario in un oggetto con accesso attributo se Generator se lo aspetta
            # (molti codici lo fanno per convenienza, es. h.num_mels invece di h['num_mels'])
            class AttrDict(dict):
                def __init__(self, *args, **kwargs):
                    super(AttrDict, self).__init__(*args, **kwargs)
                    self.__dict__ = self

            h = AttrDict(data)
            print(f"  Configurazione caricata da config.json.")
        elif os.path.exists(config_path_yml):
            print(f"  config.json non trovato, tento con args.yml...")
            with open(config_path_yml, "r") as f:
                # OmegaConf.load potrebbe restituire un oggetto OmegaConf,
                # la classe Generator potrebbe aspettarsi un dizionario o un oggetto AttrDict.
                # Assumiamo che OmegaConf.load() sia stato usato prima perché la classe Generator
                # nel tuo codice originale lo gestiva. Se usi un config.json standard,
                # OmegaConf non è necessario qui.
                # Per coerenza con il caricamento di config.json, usiamo yaml.safe_load
                # e poi lo convertiamo in AttrDict.
                data = yaml.safe_load(f)
                if isinstance(data, dict):
                    h = AttrDict(data)
                else:  # Se OmegaConf restituisce un oggetto custom
                    h = data  # Lascia che la classe Generator lo gestisca
                print(f"  Configurazione caricata da args.yml.")
        else:
            raise FileNotFoundError(
                f"Né 'config.json' né 'args.yml' trovati nella directory del vocoder: {ckpt_vocoder_dir}"
            )

        if h is None:
            raise ValueError("Impossibile caricare la configurazione del vocoder.")

        # Stampa alcuni parametri chiave per verifica
        print(f"  Parametri Vocoder (da config): SR={getattr(h, 'sampling_rate', 'N/A')}, "
              f"NumMels={getattr(h, 'num_mels', 'N/A')}, HopSize={getattr(h, 'hop_size', 'N/A')}")

        # --- Inizializzazione Architettura Generatore ---
        # Assumiamo che la classe del generatore nel tuo models.py si chiami 'Generator'
        # e che prenda 'h' (gli iperparametri) come input.
        # Nel tuo codice originale, chiamavi BigVGAN(vocoder_args), che è probabilmente la classe Generator.
        self.generator = BigVGAN(h).to(self.device)
        print(f"  Architettura Generatore ({self.generator.__class__.__name__}) inizializzata.")

        # --- Caricamento Pesi Generatore ---
        # Cerca i nomi comuni per i pesi del generatore
        possible_weights_names = [
            "generator.pth.tar",  # Se hai scaricato .tar e lo hai solo messo lì senza estrarre
            "generator.pt",  # Nome comune dopo estrazione o per checkpoint diretti
            "generator.pth",  # Altro nome comune
            "g_02500000",  # Esempio da alcuni repo (senza estensione)
            "g_02500000.pth",
            "bigvgan_generator.pt",  # Con estensione
            "best_netG.pt"  # Il nome usato nel tuo codice originale
        ]

        weights_path = None
        for name in possible_weights_names:
            path_candidate = os.path.join(ckpt_vocoder_dir, name)
            if os.path.exists(path_candidate):
                weights_path = path_candidate
                # Se è un .tar, potremmo doverlo estrarre o caricare diversamente.
                # Per ora, assumiamo che se si chiama .tar, contiene un file di pesi al suo interno
                # o che torch.load possa gestirlo (improbabile per .tar direttamente).
                # È meglio estrarre il .tar manualmente prima e puntare al file .pt/.pth interno.
                if name.endswith(".pth.tar"):
                    print(
                        f"  ATTENZIONE: Trovato file .pth.tar ('{name}'). Assicurati che sia il file di pesi corretto o estrailo.")
                    # Qui potresti aggiungere logica per estrarre se necessario,
                    # ma è meglio farlo esternamente. Se torch.load fallisce, questo è un indizio.
                break

        if weights_path is None:
            raise FileNotFoundError(
                f"Nessun file di pesi del generatore riconosciuto trovato in {ckpt_vocoder_dir}."
                f" Nomi cercati: {possible_weights_names}"
            )

        print(f"  Caricamento pesi del generatore da: {weights_path}")
        # Carica i pesi, prova prima a caricare direttamente sul device se possibile
        # per risparmiare memoria CPU, altrimenti map_location='cpu' e poi .to(device)
        try:
            # Tenta di caricare direttamente sul device target
            checkpoint = torch.load(weights_path, map_location=self.device)
        except RuntimeError as e_load_device:
            print(
                f"    WARN: Fallito caricamento diretto su {self.device} ({e_load_device}). Tento con map_location='cpu'...")
            checkpoint = torch.load(weights_path, map_location='cpu')
            # self.generator già su self.device, i pesi verranno spostati da load_state_dict

        # Logica per estrarre lo state_dict corretto dal checkpoint
        state_dict_generator = None
        if isinstance(checkpoint, dict):
            if 'generator' in checkpoint:  # Formato comune per checkpoint che includono anche discriminatori, optim, ecc.
                state_dict_generator = checkpoint['generator']
                print("    Trovata chiave 'generator' nel checkpoint.")
            elif 'model' in checkpoint:  # Altro formato comune
                state_dict_generator = checkpoint['model']
                print("    Trovata chiave 'model' nel checkpoint.")
            elif 'state_dict' in checkpoint:  # Potrebbe essere un checkpoint PyTorch Lightning
                # Filtra solo i pesi del generatore se sono prefissati
                state_dict_generator = {
                    k.replace("generator.", ""): v
                    for k, v in checkpoint['state_dict'].items()
                    if k.startswith("generator.")
                }
                if not state_dict_generator:  # Se nessun prefisso 'generator.'
                    print(
                        "    Trovata chiave 'state_dict', ma nessun prefisso 'generator.'. Assumo sia lo state_dict del generatore.")
                    state_dict_generator = checkpoint['state_dict']  # Rischioso, potrebbe contenere altro
                else:
                    print("    Trovata chiave 'state_dict' e filtrati i pesi per 'generator.'.")
            else:
                # Assume che l'intero dizionario 'checkpoint' sia lo state_dict del generatore
                state_dict_generator = checkpoint
                print(
                    "    Nessuna chiave standard ('generator', 'model', 'state_dict') trovata. Assumo che il checkpoint sia lo state_dict del generatore.")
        else:
            # Assume che l'oggetto 'checkpoint' sia direttamente lo state_dict (meno comune per i file)
            state_dict_generator = checkpoint
            print("    Il checkpoint non è un dizionario. Assumo sia direttamente lo state_dict del generatore.")

        if state_dict_generator is None:
            raise ValueError(f"Impossibile estrarre lo state_dict del generatore dal file: {weights_path}")

        # In VocoderBigVGAN.__init__
        try:
            missing_keys, unexpected_keys = self.generator.load_state_dict(state_dict_generator, strict=False)
            if missing_keys:
                print(
                    f"    ATTENZIONE: Chiavi mancanti durante il caricamento dello state_dict del generatore: {missing_keys}")
            if unexpected_keys:
                print(
                    f"    ATTENZIONE: Chiavi inattese durante il caricamento dello state_dict del generatore: {unexpected_keys}")
            print("  Pesi del generatore caricati con strict=False.")
        except Exception as e_load_strict_false:
            print(f"    ERRORE durante il caricamento con strict=False: {e_load_strict_false}")
            raise

        self.generator.eval()
        self.generator.remove_weight_norm()  # Molti modelli GAN usano weight norm in training
        print("VocoderBigVGAN inizializzato e pronto.")

    def vocode(self, mel_spectrogram_input):  # Rinominato per chiarezza
        # mel_spectrogram_input deve essere il log-mel normalizzato corretto
        with torch.no_grad():
            if isinstance(mel_spectrogram_input, np.ndarray):
                # Converte da (n_mels, T) a (1, n_mels, T) per il batch
                mel_tensor = torch.from_numpy(mel_spectrogram_input).unsqueeze(0)
            elif isinstance(mel_spectrogram_input, torch.Tensor):
                mel_tensor = mel_spectrogram_input
                if len(mel_tensor.shape) == 2:  # (n_mels, T)
                    mel_tensor = mel_tensor.unsqueeze(0)  # (1, n_mels, T)
            else:
                raise TypeError(
                    f"Input del vocoder deve essere NumPy array o PyTorch tensor, ricevuto {type(mel_spectrogram_input)}")

            # Assicurati che sia float32 e sul device corretto
            mel_tensor = mel_tensor.to(dtype=torch.float32, device=self.device)

            # Il generatore si aspetta (B, n_mels, T_frames)
            # Se il tuo mel_tensor è (B, T_frames, n_mels), devi trasporlo:
            # Esempio: if mel_tensor.shape[1] != self.generator.h.num_mels and mel_tensor.shape[2] == self.generator.h.num_mels:
            #              mel_tensor = mel_tensor.transpose(1, 2)
            # Verifica la shape attesa dal tuo Generator. La maggior parte dei vocoder si aspetta (B, n_mels, T)

            # Stampa shape per debug
            # print(f"    DEBUG Vocoder: Input mel shape: {mel_tensor.shape}")
            # print(f"    DEBUG Vocoder: Atteso num_mels dal config: {getattr(self.generator.h, 'num_mels', 'N/A')}")

            # Assicurati che num_mels corrisponda
            expected_n_mels = getattr(self.generator.h, 'num_mels', None)
            if expected_n_mels is not None and mel_tensor.shape[1] != expected_n_mels:
                raise ValueError(
                    f"Shape mismatch per n_mels: input ha {mel_tensor.shape[1]}, vocoder si aspetta {expected_n_mels}")

            waveform_output = self.generator(mel_tensor)  # (B, 1, T_samples)

            # Rimuovi la dimensione del canale audio (se è 1) e la dimensione del batch, poi sposta su CPU e NumPy
            waveform_output = waveform_output.squeeze(1).squeeze(0).cpu().numpy()
            return waveform_output

    # __call__ è utile per rendere l'oggetto chiamabile come una funzione
    def __call__(self, mel_spectrogram_input):
        return self.vocode(mel_spectrogram_input)
