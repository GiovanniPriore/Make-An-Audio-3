from preprocess.NAT_mel import MelNet
import os
from tqdm import tqdm
from glob import glob
import math
import pandas as pd
import argparse
from argparse import Namespace
import math
import audioread
from tqdm.contrib.concurrent import process_map
import torch
import torch.nn as nn
import torchaudio
import numpy as np
from torch.distributed import init_process_group
from torch.utils.data import Dataset,DataLoader,DistributedSampler
import torch.multiprocessing as mp
import json


class tsv_dataset(Dataset):
    def __init__(self,tsv_path,sr,mode='none',hop_size = None,target_mel_length = None) -> None:
        super().__init__()
        if os.path.isdir(tsv_path):
            files = glob(os.path.join(tsv_path,'*.tsv'))
            df = pd.concat([pd.read_csv(file,sep='\t') for file in files])
        else:
            df = pd.read_csv(tsv_path,sep='\t')
        self.audio_paths = []
        self.sr = sr
        self.mode = mode
        self.target_mel_length = target_mel_length
        self.hop_size = hop_size
        for t in tqdm(df.itertuples()):
            self.audio_paths.append(getattr(t,'audio_path'))

    def __len__(self):
        return len(self.audio_paths)

    def pad_wav(self,wav):
        # wav should be in shape(1,wav_len)
        wav_length = wav.shape[-1]
        assert wav_length > 100, "wav is too short, %s" % wav_length
        segment_length = (self.target_mel_length + 1) * self.hop_size  # final mel will crop the last mel, mel = mel[:,:-1]
        if segment_length is None or wav_length == segment_length:
            return wav
        elif wav_length > segment_length:
            return wav[:,:segment_length]
        elif wav_length < segment_length:
            temp_wav = torch.zeros((1, segment_length),dtype=torch.float32)
            temp_wav[:, :wav_length] = wav
        return temp_wav

    def __getitem__(self, index):
        audio_path = self.audio_paths[index]
        wav, orisr = torchaudio.load(audio_path)
        if wav.shape[0] != 1: # stereo to mono  (2,wav_len) -> (1,wav_len)
            wav = wav.mean(0,keepdim=True)
        wav = torchaudio.functional.resample(wav, orig_freq=orisr, new_freq=self.sr)
        if self.mode == 'pad':
            assert self.target_mel_length is not None
            wav = self.pad_wav(wav)
        return audio_path,wav

def process_audio_by_tsv(rank,args):
    if args.num_gpus > 1:
        init_process_group(backend=args.dist_config['dist_backend'], init_method=args.dist_config['dist_url'],
                            world_size=args.dist_config['world_size'] * args.num_gpus, rank=rank)
    
    sr = args.audio_sample_rate
    dataset = tsv_dataset(args.tsv_path,sr = sr,mode=args.mode,hop_size=args.hop_size,target_mel_length=args.batch_max_length)
    sampler = DistributedSampler(dataset,shuffle=False) if args.num_gpus > 1 else None
    # batch_size must == 1,since wav_len is not equal
    loader = DataLoader(dataset, sampler=sampler,batch_size=1, num_workers=16,drop_last=False)

    device = torch.device('cuda:{:d}'.format(rank))
    mel_net = MelNet(args.__dict__)
    mel_net.to(device)
    # if args.num_gpus > 1: # RuntimeError: DistributedDataParallel is not needed when a module doesn't have any parameter that requires a gradient.
    #     mel_net = DistributedDataParallel(mel_net, device_ids=[rank]).to(device)
    root = args.save_path
    loader = tqdm(loader) if rank == 0 else loader
    for batch in loader:
        audio_paths,wavs = batch
        wavs = wavs.to(device)
        if args.save_resample:               
            for audio_path,wav in zip(audio_paths,wavs):
                psplits = audio_path.split('/')
                wav_name = psplits[-1]
                # save resample
                resample_root,resample_name = root+f'_{sr}',wav_name[:-4]+'_audio.npy'
                resample_dir_name = os.path.join(resample_root,*psplits[1:-1])
                resample_path = os.path.join(resample_dir_name,resample_name)
                os.makedirs(resample_dir_name,exist_ok=True)
                np.save(resample_path,wav.cpu().numpy().squeeze(0))  

        if args.save_mel:
            mode = args.mode
            batch_max_length = args.batch_max_length

            for audio_path,wav in zip(audio_paths,wavs):
                psplits = audio_path.split('/')
                wav_name = psplits[-1]
                mel_root,mel_name = root,wav_name[:-4]+'_mel.npy'
                mel_dir_name = os.path.join(mel_root,f'mel{mode}{sr}',*psplits[1:-1])
                mel_path = os.path.join(mel_dir_name,mel_name)
                if not os.path.exists(mel_path):
                    mel_spec = mel_net(wav).cpu().numpy().squeeze(0) # (mel_bins,mel_len) 
                    if mel_spec.shape[1] <= batch_max_length:
                        if mode == 'tile': # pad is done in dataset as pad wav
                            n_repeat = math.ceil((batch_max_length + 1) / mel_spec.shape[1])
                            mel_spec = np.tile(mel_spec,reps=(1,n_repeat))
                        elif mode == 'none' or mode == 'pad':
                            pass
                        else:
                            raise ValueError(f'mode:{mode} is not supported')
                    mel_spec = mel_spec[:,:batch_max_length]
                    os.makedirs(mel_dir_name,exist_ok=True)
                    np.save(mel_path,mel_spec)      


def split_list(i_list,num):
    each_num = math.ceil(i_list / num)
    result = []
    for i in range(num):
        s = each_num * i
        e = (each_num * (i+1))
        result.append(i_list[s:e])
    return result


def drop_bad_wav(item):
    index,path = item
    try:
        with audioread.audio_open(path) as f:
            totalsec = f.duration
            if totalsec < 0.1:
                return index # index
    except:
        print(f"corrupted wav:{path}")
        return index
    return False 

def drop_bad_wavs(tsv_path):# 'audioset.csv'
    df = pd.read_csv(tsv_path,sep='\t')
    item_list = []
    for item in tqdm(df.itertuples()):
        item_list.append((item[0],getattr(item,'audio_path')))

    r = process_map(drop_bad_wav,item_list,max_workers=16,chunksize=16)
    bad_indices = list(filter(lambda x:x!= False,r))
        
    print(bad_indices)
    with open('bad_wavs.json','w') as f:
        x = [item_list[i] for i in bad_indices]
        json.dump(x,f)
    df = df.drop(bad_indices,axis=0)
    df.to_csv(tsv_path,sep='\t',index=False)

def addmel2tsv(save_dir,tsv_path):
    df = pd.read_csv(tsv_path,sep='\t')
    mels = glob(f'{save_dir}/mel{args.mode}{args.audio_sample_rate}/**/*_mel.npy',recursive=True)
    name2mel,idx2name,idx2mel = {},{},{}
    for mel in mels:
        bn = os.path.basename(mel)[:-8]# remove _mel.npy
        name2mel[bn] = mel
    for t in df.itertuples():
        idx = int(t[0])
        bn = os.path.basename(getattr(t,'audio_path'))[:-4]
        idx2name[idx] = bn
    for k,v in idx2name.items():
        idx2mel[k] = name2mel[v]
    df['mel_path'] = df.index.map(idx2mel)
    df.to_csv(tsv_path,sep='\t',index=False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--tsv_path",type=str)
    parser.add_argument( "--num_gpus",type=int,default=1)
    parser.add_argument( "--max_duration",type=int,default=30)
    return parser.parse_args()


if __name__ == '__main__':
    pargs = parse_args()  # Legge --tsv_path, --num_gpus, --max_duration da CLI

    # --- INIZIO PARAMETRI PERSONALIZZATI PER KAGGLE (SINGOLO FILE) ---
    # Questi valori sovrascrivono o forniscono i default per i parametri usati sotto.

    # 1. Percorso al TSV che hai creato (contiene UNA riga per il tuo file audio)
    #    Questo sovrascrive pargs.tsv_path se fornito da CLI, o il default di pargs se non fornito.
    current_tsv_path = "/kaggle/working/mio_singolo_file_manifest.tsv"  # ASSICURATI CHE QUESTO ESISTA E SIA CORRETTO

    # 2. Numero di GPU (su Kaggle di solito 1)
    #    Usa il valore da CLI se fornito, altrimenti default a 1.
    current_num_gpus = pargs.num_gpus if hasattr(pargs, 'num_gpus') and pargs.num_gpus is not None else 1
    if current_num_gpus is None: current_num_gpus = 1  # Ulteriore fallback

    # 3. Durata target per i segmenti Mel (in secondi)
    #    Questo sovrascrive pargs.max_duration (che ha default 30 nello script originale)
    #    Allinea questo a TARGET_AUDIO_MAX_DURATION_SECONDS dalla tua configurazione principale.
    TARGET_MEL_DURATION_SEC = 10

    # 4. Sample Rate Target per i Mel (DEVE ESSERE CONSISTENTE CON IL VAE)
    TARGET_MEL_SR_HZ = 22050

    # 5. Hop Size per il calcolo dei Mel (comune, usato per calcolare batch_max_length)
    TARGET_MEL_HOP_SIZE = 256

    # 6. Modalità di processamento Mel ('pad', 'none', 'tile')
    #    'pad' assicura che tutti i Mel abbiano batch_max_length.
    TARGET_MEL_MODE = 'pad'

    # 7. Directory base dove verranno salvati i Mel.
    #    Lo script creerà sottocartelle come mel{TARGET_MEL_MODE}{TARGET_MEL_SR_HZ}
    MEL_SAVE_ROOT_PATH = "/kaggle/working/my_single_video_preprocessed/generated_mels_script_output"  # Deve corrispondere alla FASE 0

    # --- FINE PARAMETRI PERSONALIZZATI ---

    # Esegui drop_bad_wavs sul nostro TSV specifico
    # (Attenzione: questo modifica il file TSV sul posto se trova file corrotti)
    print(f"Controllo file audio in: {current_tsv_path}")
    if os.path.exists(current_tsv_path):
        if os.path.isdir(current_tsv_path):  # Logica originale per directory di TSV
            files = glob(os.path.join(current_tsv_path, '*.tsv'))
            for file_loop_var in files:  # Rinominata 'file'
                drop_bad_wavs(file_loop_var)
        else:  # Il nostro caso: singolo file TSV
            drop_bad_wavs(current_tsv_path)
    else:
        print(f"ERRORE: File TSV '{current_tsv_path}' non trovato!")
        exit()  # Esci se il TSV non c'è

    # Calcola batch_max_length basato sui nostri parametri target
    # Questo è il numero di frame Mel per la durata target
    current_batch_max_length = int(TARGET_MEL_DURATION_SEC * (TARGET_MEL_SR_HZ / TARGET_MEL_HOP_SIZE))

    args_dict = {
        'audio_sample_rate': TARGET_MEL_SR_HZ,
        'audio_num_mel_bins': 80,  # Valore comune, verifica se il tuo modello si aspetta altro
        'fft_size': 1024,  # Valore comune
        'win_size': 1024,  # Valore comune
        'hop_size': TARGET_MEL_HOP_SIZE,
        'fmin': 0,
        'fmax': TARGET_MEL_SR_HZ // 2,  # Frequenza di Nyquist
        'batch_max_length': current_batch_max_length,
        'tsv_path': current_tsv_path,
        'num_gpus': current_num_gpus,
        'mode': TARGET_MEL_MODE,
        'save_resample': False,  # Di solito non serve per l'inferenza
        'save_mel': True,  # Vogliamo salvare i Mel
        'save_path': MEL_SAVE_ROOT_PATH,
    }
    os.makedirs(MEL_SAVE_ROOT_PATH, exist_ok=True)  # Assicura che la directory base esista

    print(f"DEBUG (mel_spec.py): Argomenti effettivi prima di Namespace: {args_dict}")
    args_namespace = Namespace(**args_dict)

    args_namespace.dist_config = {  # Config di default per la distribuzione
        "dist_backend": "nccl",
        "dist_url": "tcp://localhost:54189",
        "world_size": 1  # Assumiamo 1 GPU per il setup di default di dist_config
    }
    # Se current_num_gpus è > 1 e world_size deve essere diverso, questa parte potrebbe necessitare aggiustamenti,
    # ma per num_gpus=1, la logica mp.spawn non verrà attivata.

    if args_namespace.num_gpus > 1:
        # Assicurati che args_namespace.dist_config['world_size'] sia corretto per multi-GPU
        # args_namespace.dist_config['world_size'] = args_namespace.num_gpus # Esempio
        mp.spawn(process_audio_by_tsv, nprocs=args_namespace.num_gpus, args=(args_namespace,))
    else:
        process_audio_by_tsv(0, args=args_namespace)  # Passa l'oggetto Namespace

    print("Processo generazione Mel completato.")

    # Gestione di addmel2tsv:
    # La funzione addmel2tsv nello script originale potrebbe usare una variabile globale 'args'.
    # Per sicurezza, proviamo a renderla disponibile o passarla.
    try:
        # Tentativo 1: definire args globale (un po' un hack)
        # globals()['args'] = args_namespace
        # Tentativo 2: se addmel2tsv potesse prendere args come parametro (migliore ma richiede modifica di addmel2tsv)
        # addmel2tsv(args_namespace.save_path, args_namespace.tsv_path, args_namespace=args_namespace)

        # Per ora, assumiamo che addmel2tsv possa trovare 'args' globalmente come nello script originale
        # e che il nostro args_namespace sia sufficiente se per caso lo usa.
        # Altrimenti, il TSV non verrà aggiornato, ma i Mel dovrebbero essere stati salvati.
        # Lo script originale passava args.mode e args.audio_sample_rate a addmel2tsv implicitamente.
        # Dobbiamo assicurarci che la funzione addmel2tsv possa accedere a questi valori.
        # Un modo è definire 'args' globalmente prima di chiamarla.
        args = args_namespace  # Rendi l'oggetto namespace accessibile come 'args' globale
        addmel2tsv(args_namespace.save_path, args_namespace.tsv_path)
        print("Funzione addmel2tsv completata (tentativo di aggiornare TSV).")
    except Exception as e_addmel:
        print(f"ATTENZIONE: Errore durante l'esecuzione di addmel2tsv: {e_addmel}")
        print(
            "I file Mel potrebbero essere stati generati, ma il TSV potrebbe non essere stato aggiornato con mel_path.")
        print("Questo di solito non è un problema per ottenere solo i file Mel.")

    print("Script mel_spec.py terminato.")
    
