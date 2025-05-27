# Contenuto per /kaggle/working/Make-An-Audio-3/preprocess/mel_spec.py
# Modificato per hardcodare i parametri per il test con singolo file.

from preprocess.NAT_mel import MelNet  # Assicurati che NAT_mel.py sia in preprocess/
import os
from tqdm import tqdm
from glob import glob
import math
import pandas as pd
import argparse
from argparse import Namespace
# import math # Già importato
import audioread
from tqdm.contrib.concurrent import process_map
import torch
import torch.nn as nn
import torchaudio
import numpy as np
from torch.distributed import init_process_group
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.multiprocessing as mp
import json


class tsv_dataset(Dataset):
    def __init__(self, tsv_path, sr, mode='none', hop_size=None, target_mel_length=None) -> None:
        super().__init__()
        if os.path.isdir(tsv_path):
            files = glob(os.path.join(tsv_path, '*.tsv'))
            df = pd.concat([pd.read_csv(file, sep='\t') for file in files])
        else:
            df = pd.read_csv(tsv_path, sep='\t')
        self.audio_paths = []
        self.sr = sr
        self.mode = mode
        self.target_mel_length = target_mel_length
        self.hop_size = hop_size
        for t in tqdm(df.itertuples(), desc="Loading audio paths from TSV"):
            self.audio_paths.append(getattr(t, 'audio_path'))

    def __len__(self):
        return len(self.audio_paths)

    def pad_wav(self, wav):
        wav_length = wav.shape[-1]
        if wav_length <= 100:  # Aggiunto controllo per wav troppo corti prima dell'assert
            print(f"Warning: wav is very short, length: {wav_length}. Padding to a minimum length for processing.")
            min_len_for_processing = (self.hop_size * 5)  # es. 5 frame mel
            if wav_length < min_len_for_processing:
                padding = torch.zeros((1, min_len_for_processing - wav_length), dtype=wav.dtype, device=wav.device)
                wav = torch.cat((wav, padding), dim=1)
                wav_length = wav.shape[-1]
        assert wav_length > 100, "wav is too short, %s" % wav_length

        segment_length = (self.target_mel_length + 1) * self.hop_size
        if segment_length is None or wav_length == segment_length:
            return wav
        elif wav_length > segment_length:
            return wav[:, :segment_length]
        elif wav_length < segment_length:
            temp_wav = torch.zeros((1, segment_length), dtype=torch.float32, device=wav.device)
            temp_wav[:, :wav_length] = wav
            return temp_wav  # Aggiunto return
        return wav  # Fallback

    def __getitem__(self, index):
        audio_path = self.audio_paths[index]
        try:
            wav, orisr = torchaudio.load(audio_path)
            if wav.shape[0] != 1:
                wav = wav.mean(0, keepdim=True)
            if orisr != self.sr:  # Resample solo se necessario
                wav = torchaudio.functional.resample(wav, orig_freq=orisr, new_freq=self.sr)
            if self.mode == 'pad':
                assert self.target_mel_length is not None and self.hop_size is not None, "target_mel_length and hop_size must be set for pad mode"
                wav = self.pad_wav(wav)
            return audio_path, wav
        except Exception as e:
            print(f"Error loading/processing {audio_path}: {e}")
            # Ritorna un placeholder o None per segnalare l'errore
            # Per un singolo file, un errore qui è problematico.
            # Potremmo ritornare un tensore di zeri della forma attesa per evitare che il DataLoader si blocchi,
            # ma il Mel generato sarà inutile.
            dummy_len = (
                                    self.target_mel_length + 1) * self.hop_size if self.mode == 'pad' and self.target_mel_length and self.hop_size else self.sr  # 1 secondo di zeri
            return audio_path, torch.zeros((1, dummy_len))


def process_audio_by_tsv(rank, args_namespace):  # Rinominato args a args_namespace per chiarezza
    if args_namespace.num_gpus > 1:
        init_process_group(backend=args_namespace.dist_config['dist_backend'],
                           init_method=args_namespace.dist_config['dist_url'],
                           world_size=args_namespace.dist_config['world_size'] * args_namespace.num_gpus, rank=rank)

    sr = args_namespace.audio_sample_rate
    print(
        f"Process {rank}: Loading dataset from TSV: {args_namespace.tsv_path} with SR={sr}, mode={args_namespace.mode}")
    dataset = tsv_dataset(args_namespace.tsv_path, sr=sr, mode=args_namespace.mode, hop_size=args_namespace.hop_size,
                          target_mel_length=args_namespace.batch_max_length)

    if len(dataset) == 0:
        print(f"Process {rank}: Dataset is empty. No files to process from {args_namespace.tsv_path}.")
        return

    sampler = DistributedSampler(dataset, shuffle=False) if args_namespace.num_gpus > 1 else None
    loader = DataLoader(dataset, sampler=sampler, batch_size=1, num_workers=1,
                        drop_last=False)  # num_workers a 1 o 0 per debug

    device = torch.device('cuda:{:d}'.format(rank))
    print(f"Process {rank}: Using device: {device}")
    try:
        mel_net = MelNet(args_namespace.__dict__)  # MelNet potrebbe avere bisogno di specifici parametri da args
        mel_net.to(device)
    except Exception as e:
        print(f"Process {rank}: Error initializing MelNet: {e}")
        print(f"Process {rank}: Args passed to MelNet: {args_namespace.__dict__}")
        return

    root_save_path = args_namespace.save_path  # Questa è la directory base, es. /kaggle/working/.../generated_mels_script_output

    # La barra di progresso solo per il rank 0 se multi-GPU
    iterable_loader = tqdm(loader, desc=f"Process {rank} Mel Generation") if rank == 0 else loader

    for batch_idx, batch in enumerate(iterable_loader):
        audio_paths, wavs = batch

        if wavs is None or wavs.nelement() == 0:  # Controllo se wav è valido
            print(f"Process {rank}, Batch {batch_idx}: Skipping corrupted or empty audio {audio_paths[0]}")
            continue

        wavs = wavs.to(device)
        # print(f"Process {rank}, Batch {batch_idx}: Processing {audio_paths[0]}, wav shape on device: {wavs.shape}")

        if args_namespace.save_mel:
            mode = args_namespace.mode
            batch_max_length = args_namespace.batch_max_length  # Lunghezza Mel in frame

            for audio_path_str, wav_tensor in zip(audio_paths, wavs):  # audio_path è una tupla se batch_size>1
                current_audio_path = Path(audio_path_str)
                audio_filename_stem = current_audio_path.stem  # es. "mio_audio_estratto"

                # Costruzione del percorso di output: root_save_path / "mel{mode}{sr}" / (parti del path originale) / nomefile_mel.npy
                # Per un singolo file, con audio_path assoluto, le "parti del path originale" possono essere lunghe.
                # Semplifichiamo per il singolo file: salviamo direttamente sotto mel{mode}{sr} usando YOUR_SAMPLE_NAME_ID

                # Questo nome di output deve essere coerente con come lo cerchiamo dopo
                output_mel_subdir = os.path.join(root_save_path, f'mel{mode}{sr}')

                # Per il NOME del file Mel, usiamo YOUR_SAMPLE_NAME_ID definito nel TSV
                # Lo script originale usa getattr(t,'name') per il nome base se esistesse nel TSV.
                # Il nostro TSV ha 'name', quindi dovremmo leggerlo o passarlo.
                # Per ora, usiamo audio_filename_stem, che viene da audio_path.
                # Il TSV che creiamo ha YOUR_SAMPLE_NAME_ID come 'name'.
                # Il Path(actual_audio_file_for_mel_tsv).stem darà il nome del file audio effettivo.

                mel_filename = f"{audio_filename_stem}_mel.npy"  # es. "mio_audio_estratto_mel.npy"
                final_mel_path = os.path.join(output_mel_subdir, mel_filename)

                # Assicurati che la directory di output esista
                os.makedirs(output_mel_subdir, exist_ok=True)

                if not os.path.exists(final_mel_path):
                    try:
                        # print(f"Process {rank}: Generating Mel for {current_audio_path} -> {final_mel_path}")
                        # print(f"Process {rank}: Wav shape: {wav_tensor.shape}, target mel length (frames): {batch_max_length}")
                        mel_spec = mel_net(wav_tensor).cpu().numpy().squeeze(0)  # (mel_bins,mel_len)

                        # La logica originale di padding/tiling/truncating
                        if mel_spec.shape[1] <= batch_max_length:
                            if mode == 'tile':
                                if mel_spec.shape[1] == 0: print(
                                    f"Warning: mel_spec is empty for {current_audio_path}"); continue
                                n_repeat = math.ceil((batch_max_length + 1) / mel_spec.shape[1])
                                mel_spec = np.tile(mel_spec, reps=(1, n_repeat))
                            # 'pad' mode: il padding dell'audio è già stato fatto nel __getitem__ del dataset
                            #             per assicurare che il mel risultante sia <= batch_max_length
                            # 'none' mode: nessun padding/tiling qui, solo troncamento sotto

                        mel_spec = mel_spec[:, :batch_max_length]  # Troncamento finale alla lunghezza massima
                        # print(f"Process {rank}: Generated Mel shape: {mel_spec.shape}")

                        np.save(final_mel_path, mel_spec)
                        if rank == 0: print(f"Saved Mel: {final_mel_path}")
                    except Exception as e_mel:
                        print(f"Process {rank}: Error generating/saving Mel for {current_audio_path}: {e_mel}")
                # else:
                # if rank == 0: print(f"Mel already exists: {final_mel_path}")


# Le funzioni split_list, drop_bad_wav, drop_bad_wavs, addmel2tsv non sono
# strettamente necessarie per processare un singolo file se il TSV è già corretto
# e non abbiamo bisogno di aggiornare il TSV con mel_path per il test.
# Le lascio commentate per ora per semplificare.

# def split_list(i_list,num): ...
# def drop_bad_wav(item): ...
# def drop_bad_wavs(tsv_path): ...
# def addmel2tsv(save_dir,tsv_path, args_ns): ... # Modificato per prendere args


def parse_args():  # Questa funzione è ancora usata per --num_gpus originale
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv_path", type=str, default="/kaggle/working/placeholder.tsv")  # Default non usato
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--max_duration", type=int, default=10)  # Default cambiato a 10s
    return parser.parse_args()


if __name__ == '__main__':
    pargs = parse_args()

    # --- PARAMETRI HARDCODED PER IL TEST CON SINGOLO FILE ---
    # Questi valori sono quelli che il tuo script principale (blocco unico) dovrebbe passare
    # o che corrispondono alla tua configurazione.

    current_tsv_path = "/kaggle/working/mio_singolo_file_manifest.tsv"  # DA CONFIGURARE NEL BLOCCO UNICO
    current_num_gpus = pargs.num_gpus if hasattr(pargs, 'num_gpus') and pargs.num_gpus is not None else 1
    if current_num_gpus is None: current_num_gpus = 1

    TARGET_MEL_DURATION_SEC = 10
    TARGET_MEL_SR_HZ = 22050
    TARGET_MEL_HOP_SIZE = 256
    TARGET_MEL_MODE = 'pad'
    MEL_SAVE_ROOT_PATH = "/kaggle/working/my_single_video_preprocessed/generated_mels_script_output"  # DA CONFIGURARE NEL BLOCCO UNICO

    # --- FINE PARAMETRI HARDCODED ---

    print(f"MEL_SPEC.PY (MODIFICATO): Inizio esecuzione con i seguenti parametri hardcoded:")
    print(f"  TSV Path: {current_tsv_path}")
    print(f"  Num GPUs: {current_num_gpus}")
    print(f"  Target Duration: {TARGET_MEL_DURATION_SEC}s")
    print(f"  Target SR: {TARGET_MEL_SR_HZ}Hz")
    print(f"  Target Hop Size: {TARGET_MEL_HOP_SIZE}")
    print(f"  Target Mode: {TARGET_MEL_MODE}")
    print(f"  Save Root Path: {MEL_SAVE_ROOT_PATH}")

    if not os.path.exists(current_tsv_path):
        print(f"ERRORE (mel_spec.py): File TSV '{current_tsv_path}' non trovato! Crea questo file prima.")
        exit()

    current_batch_max_length = int(TARGET_MEL_DURATION_SEC * (TARGET_MEL_SR_HZ / TARGET_MEL_HOP_SIZE))

    args_dict = {
        'audio_sample_rate': TARGET_MEL_SR_HZ,
        'audio_num_mel_bins': 80,
        'fft_size': 1024,
        'win_size': 1024,
        'hop_size': TARGET_MEL_HOP_SIZE,
        'fmin': 0,
        'fmax': TARGET_MEL_SR_HZ // 2,
        'batch_max_length': current_batch_max_length,
        'tsv_path': current_tsv_path,
        'num_gpus': current_num_gpus,
        'mode': TARGET_MEL_MODE,
        'save_resample': False,
        'save_mel': True,
        'save_path': MEL_SAVE_ROOT_PATH,
    }
    os.makedirs(MEL_SAVE_ROOT_PATH, exist_ok=True)

    print(f"DEBUG (mel_spec.py): Dizionario args effettivo: {args_dict}")
    args_namespace = Namespace(**args_dict)

    args_namespace.dist_config = {
        "dist_backend": "nccl",
        "dist_url": "tcp://localhost:54189",
        "world_size": 1
    }

    if args_namespace.num_gpus > 1:
        mp.spawn(process_audio_by_tsv, nprocs=args_namespace.num_gpus, args=(args_namespace,))
    else:
        process_audio_by_tsv(0, args=args_namespace)

    print("Processo generazione Mel completato da mel_spec.py.")

    # addmel2tsv è commentato perché non strettamente necessario per ottenere solo il file Mel
    # e potrebbe richiedere che 'args' sia un oggetto globale con certi attributi.
    # try:
    #     # globals()['args'] = args_namespace # Rendi args_namespace disponibile come 'args' globale
    #     # addmel2tsv(args_namespace.save_path, args_namespace.tsv_path) # Potrebbe fallire se 'args' non è come si aspetta
    #     print("Skipping addmel2tsv per il test con singolo file.")
    # except Exception as e_addmel:
    #     print(f"ATTENZIONE: Errore durante l'esecuzione di addmel2tsv: {e_addmel}")

    print("Script mel_spec.py (modificato) terminato.")