import argparse, os, sys, glob
import pathlib

directory = pathlib.Path(os.getcwd())
print(directory)
sys.path.append(str(directory))  # Questo è buono se lo script è nella root del progetto

import torch
import numpy as np
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
# from ldm.models.diffusion.ddim import DDIMSampler # Non usato in questo script specifico
# from ldm.models.diffusion.plms import PLMSSampler # Non usato in questo script specifico
import random, math  # librosa non è usato qui, ma lo era nel tuo script di estrazione
from vocoder.bigvgan.models import VocoderBigVGAN
import soundfile
from pathlib import Path
from tqdm import tqdm

# All'inizio dello script, dopo gli import
PATH_TO_NVIDIA_VOCODER_DIR_IN_V2A = "/kaggle/input/newvocoder/bigvgan_v2_24khz_100band_256x/" # << MODIFICA
NVIDIA_VOCODER_CONFIG_JSON_IN_V2A = os.path.join(PATH_TO_NVIDIA_VOCODER_DIR_IN_V2A, "config.json")

# Variabili globali per i parametri del vocoder NVIDIA (saranno popolate)
NV_SR, NV_N_FFT, NV_NUM_MELS, NV_HOP_SIZE, NV_WIN_SIZE, NV_FMIN, NV_FMAX = [None]*7


# Questa è la versione della funzione che dovrebbe gestire i None per fmax
# e usare get_int_param per le altre chiavi.
# Assicurati che sia QUESTA la versione nel tuo script.

def load_nvidia_vocoder_params_for_v2a(config_json_path):  # Rinominata nel tuo script
    global NV_SR, NV_N_FFT, NV_NUM_MELS, NV_HOP_SIZE, \
        NV_WIN_SIZE, NV_FMIN, NV_FMAX

    if not os.path.exists(config_json_path):
        print(f"ERRORE V2A: File config.json vocoder NVIDIA '{config_json_path}' non trovato.")
        return False

    import json  # Assicurati che json sia importato qui o globalmente
    with open(config_json_path, 'r') as f:
        config = json.load(f)

        print("--- CONTENUTO RAW DEL CONFIG.JSON CARICATO ---")
        print(json.dumps(config, indent=2))  # Stampa il JSON formattato
        print("-------------------------------------------")

    try:
        # Funzione helper per caricare int con fallback se None o mancante
        def get_int_param(cfg_dict, key, default_value=None, is_critical=True):
            val = cfg_dict.get(key)  # Usa .get() per evitare KeyError se la chiave manca
            if val is not None:
                try:
                    return int(val)  # Converte in int
                except (ValueError, TypeError):  # Se val non è convertibile in int
                    if default_value is not None:
                        print(
                            f"ATTENZIONE: Valore per '{key}' ('{val}') non è un intero valido, uso default: {default_value}")
                        return default_value
                    elif is_critical:
                        raise ValueError(f"Chiave critica '{key}' ha un valore non intero ('{val}') e nessun default.")
                    else:
                        return None  # O 0 per fmin
            elif default_value is not None:  # Se val è None e c'è un default
                print(f"ATTENZIONE: Chiave '{key}' non trovata o None nel config, uso default: {default_value}")
                return default_value
            elif is_critical:  # Se val è None, non c'è default, ed è critica
                raise KeyError(f"Chiave critica '{key}' mancante o None nel config e nessun default fornito.")
            else:  # Non critica, nessun default, restituisci None (o 0 per fmin)
                print(f"ATTENZIONE: Chiave opzionale '{key}' non trovata o None, sarà gestita come None/0.")
                return None

        NV_SR = get_int_param(config, 'sampling_rate', is_critical=True)
        NV_N_FFT = get_int_param(config, 'n_fft', is_critical=True)
        NV_NUM_MELS = get_int_param(config, 'num_mels', is_critical=True)
        NV_HOP_SIZE = get_int_param(config, 'hop_size', is_critical=True)
        NV_WIN_SIZE = get_int_param(config, 'win_size', is_critical=True)

        NV_FMIN = get_int_param(config, 'fmin', default_value=0, is_critical=False)
        if NV_FMIN is None: NV_FMIN = 0  # Assicura sia int

        fmax_val = config.get('fmax')  # Prendi fmax, potrebbe essere None (null) o un numero
        if fmax_val is not None:
            try:
                NV_FMAX = int(fmax_val)
            except (ValueError, TypeError):
                print(f"ATTENZIONE: 'fmax' ('{fmax_val}') non è un intero valido, calcolo come SR/2.")
                if NV_SR is None: raise ValueError("NV_SR non definito per calcolare FMAX di default.")
                NV_FMAX = NV_SR // 2
        else:  # fmax_val è None (chiave mancante o valore null)
            print(f"ATTENZIONE: 'fmax' è None/null o mancante nel config, calcolo come SR/2.")
            if NV_SR is None: raise ValueError("NV_SR non definito per calcolare FMAX di default.")
            NV_FMAX = NV_SR // 2

        # Verifica che tutti i parametri critici siano stati caricati
        critical_params = {"SR": NV_SR, "N_FFT": NV_N_FFT, "NUM_MELS": NV_NUM_MELS,
                           "HOP_SIZE": NV_HOP_SIZE, "WIN_SIZE": NV_WIN_SIZE}
        for name, val_check in critical_params.items():
            if val_check is None:
                print(f"ERRORE CRITICO POST-CARICAMENTO: Parametro '{name}' è ancora None.")
                return False

        print("\n--- Parametri Vocoder NVIDIA Caricati ---")
        # ... (stampa i parametri) ...
        return True

    except KeyError as e:  # Questo dovrebbe essere catturato da get_int_param se is_critical=True
        print(f"ERRORE CRITICO (KeyError): Chiave '{e}' mancante nel config.json e necessaria.")
        return False
    except ValueError as e_val:
        print(f"ERRORE di VALORE durante il caricamento (es. conversione int fallita): {e_val}")
        return False
    except Exception as e_load:
        print(f"ERRORE generico durante il caricamento dei parametri: {e_load}")
        # import traceback # Togli il commento per debug dettagliato se necessario
        # traceback.print_exc()
        return False

# --- FINE AGGIUNTA ---

print("\nDEBUG: Inizio blocco di caricamento parametri globali del vocoder NVIDIA.")
NVIDIA_VOCODER_CONFIG_JSON_IN_V2A = os.path.join(PATH_TO_NVIDIA_VOCODER_DIR_IN_V2A, "config.json") # Ricontrolla PATH_TO_NVIDIA_VOCODER_DIR_IN_V2A

print(f"DEBUG: Chiamata a load_nvidia_vocoder_params_for_v2a con path: {NVIDIA_VOCODER_CONFIG_JSON_IN_V2A}")
SUCCESS_LOAD_NVIDIA_PARAMS = load_nvidia_vocoder_params_for_v2a(NVIDIA_VOCODER_CONFIG_JSON_IN_V2A)
print(f"DEBUG: Valore di SUCCESS_LOAD_NVIDIA_PARAMS dopo la chiamata: {SUCCESS_LOAD_NVIDIA_PARAMS}")
print(f"DEBUG: Valore di NV_SR dopo la chiamata: {NV_SR}") # Controlla se è stato impostato

if not SUCCESS_LOAD_NVIDIA_PARAMS:
    print("ERRORE CRITICO (globale): Impossibile caricare i parametri del vocoder NVIDIA. Script terminato ORA.")
    sys.exit(1)
else:
    print("INFO (globale): Parametri Vocoder NVIDIA caricati con successo. NV_SR è impostato.")


def load_model_from_config(config, ckpt=None, verbose=True):
    # Assicurati che la modifica per weights_only=False sia qui se non è già nel ckpt
    # Il tuo ckpt principale è per il modello text-to-audio.
    # Questo script carica un checkpoint V2A separato? opt.resume lo controlla.
    # Se il formato del checkpoint V2A è simile e causa problemi con PyTorch 2.6,
    # la modifica a weights_only=False potrebbe essere necessaria anche qui.
    # Per ora, lo lasciamo com'è, ma tienilo a mente.
    model = instantiate_from_config(config.model)
    if ckpt:
        print(f"Loading model from {ckpt}")
        # --- MODIFICA QUI ---
        # Riga originale:
        # pl_sd = torch.load(ckpt, map_location="cpu")
        # Riga Modificata:
        pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        # --- FINE MODIFICA ---

        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        else:
            # A volte i checkpoint salvati direttamente (non da PyTorch Lightning) sono solo lo state_dict
            print("WARN: 'state_dict' non trovato nel checkpoint, assumo che il checkpoint sia lo state_dict stesso.")
            sd = pl_sd

        if "epoch" in pl_sd and "global_step" in pl_sd:
            print(
                f'---------------------------epoch : {pl_sd["epoch"]}, global step: {pl_sd["global_step"] // 1e3}k---------------------------')
        else:
            print("WARN: Chiavi 'epoch' o 'global_step' non trovate nel checkpoint.")

        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)
    else:
        print(f"Note: no model checkpoint (ckpt) is loaded via load_model_from_config!!!")

    model.cuda()
    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,  # Convertito a int
        help="sample rate of wav"
    )
    parser.add_argument(
        "--length",
        type=int,
        default=None,
        help="length of wav (in number of mel frames)"
    )
    parser.add_argument(
        "--test_dataset",
        default="vggsound",  # ### MODIFICA: Potresti cambiare il default a "custom" se lo usi sempre così
        help="test which dataset: vggsound/landscape/fsd50k/custom"  ### MODIFICA: Aggiunta opzione "custom"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/video2audio-samples"  # Modificato per chiarezza
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=25,  # Originale, potrebbe essere 50 o 200 per CFM
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        # const=True, default="", nargs="?", # Rimosso const=True se vuoi sempre passare un path
        help="resume from checkpoint path for the main V2A model",
        required=True  # Rendilo richiesto se carichi sempre un modello V2A
    )
    parser.add_argument(
        "-b",
        "--base",
        type=str,
        help="paths to base configs. Loaded from left-to-right.",
        required=True  # Rendilo richiesto
    )

    ### MODIFICA START: Aggiunta argomenti per il dataset custom ###
    parser.add_argument(
        "--custom_data_list_txt",
        type=str,
        default="/kaggle/working/my_v2a_test_data/my_test_list.txt",  # Default per Kaggle
        help="Path to the text file listing custom data samples (for --test_dataset custom)"
    )
    parser.add_argument(
        "--custom_visual_features_dir",
        type=str,
        default="/kaggle/working/my_v2a_test_data/visual_features/",  # Default per Kaggle
        help="Path to the directory containing custom visual features (for --test_dataset custom)"
    )
    parser.add_argument(
        "--custom_audio_mels_gt_dir",
        type=str,
        default="/kaggle/working/my_v2a_test_data/audio_mels_gt/",  # Default per Kaggle
        help="Path to the directory containing custom audio mel GTs (for --test_dataset custom)"
    )
    parser.add_argument(
        "--custom_empty_vid_path",
        type=str,
        default="/kaggle/working/my_v2a_test_data/empty_video_data/empty_vid.npz",  # Default per Kaggle
        help="Path to the custom empty_vid.npz file (for --test_dataset custom and scale != 1.0)"
    )
    ### MODIFICA END ###

    return parser.parse_args()


# ... (tutti gli import e le definizioni di funzione precedenti, inclusa
#      PATH_TO_NVIDIA_VOCODER_DIR_IN_V2A, NVIDIA_VOCODER_CONFIG_JSON_IN_V2A,
#      le variabili NV_..., load_nvidia_vocoder_params_for_v2a,
#      la gestione di TorchDynamo, load_model_from_config, parse_args) ...

# Assicurati che le variabili NV_... siano state caricate prima di main()
# SUCCESS_LOAD_NVIDIA_PARAMS = load_nvidia_vocoder_params_for_v2a(NVIDIA_VOCODER_CONFIG_JSON_IN_V2A)
# if not SUCCESS_LOAD_NVIDIA_PARAMS:
#     print("ERRORE CRITICO: Impossibile caricare i parametri del vocoder NVIDIA. Uscita.")
#     sys.exit(1)

def main():
    opt = parse_args()

    # Imposta la Sample Rate di Output basata sul Vocoder NVIDIA
    # Sovrascrive l'argomento --sample_rate se diverso, con un avviso.
    if NV_SR is None:  # Controllo extra, dovrebbe essere già stato caricato
        print("ERRORE CRITICO: Parametri Vocoder NVIDIA (NV_SR) non caricati prima di main().")
        sys.exit(1)

    OUTPUT_AUDIO_SR = NV_SR
    if opt.sample_rate != OUTPUT_AUDIO_SR:
        print(f"ATTENZIONE V2A: opt.sample_rate ({opt.sample_rate}) è stato sovrascritto da NV_SR ({OUTPUT_AUDIO_SR}). "
              f"L'output audio sarà a {OUTPUT_AUDIO_SR}Hz.")
    # opt.sample_rate = OUTPUT_AUDIO_SR # Opzionale: aggiorna l'oggetto opt se altre parti lo usano

    print("--- ARGOMENTI PARSATI (V2A SCRIPT) ---")
    for arg, value in vars(opt).items():
        print(f"  {arg}: {value}")
    print(f"  Sample Rate Effettiva per Output Audio: {OUTPUT_AUDIO_SR}")
    print("------------------------------------")

    config = OmegaConf.load(opt.base)

    if not opt.resume or not os.path.exists(opt.resume):
        print(f"ERRORE: Checkpoint V2A (--resume) non specificato o non trovato: {opt.resume}")
        sys.exit(1)
    model = load_model_from_config(config, opt.resume)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Modello principale V2A spostato su: {device}")

    os.makedirs(opt.outdir, exist_ok=True)
    print(f"Directory di output: {opt.outdir}")

    print(f"Tentativo di caricare il Vocoder NVIDIA dalla directory: {PATH_TO_NVIDIA_VOCODER_DIR_IN_V2A}")
    try:
        vocoder = VocoderBigVGAN(ckpt_vocoder_dir=PATH_TO_NVIDIA_VOCODER_DIR_IN_V2A, device=device)
        print(f"Vocoder NVIDIA caricato con successo.")
    except Exception as e_vocoder_init:
        print(f"ERRORE durante l'inizializzazione del Vocoder NVIDIA: {e_vocoder_init}")
        import traceback
        traceback.print_exc()  # Spostato qui dentro l'except
        sys.exit(1)  # Spostato qui dentro l'except

    spec_list1 = []
    video_list1 = []
    data_list1_basenames = []  # Per i nomi base

    if opt.test_dataset == 'custom':
        print(f"\nCaricamento dataset 'custom' dai path specificati:")
        # ... (stampa dei path custom come prima) ...
        if not os.path.exists(opt.custom_data_list_txt):  # ... (controlli esistenza path) ...
            sys.exit(1)
        # ... (altri controlli)

        with open(opt.custom_data_list_txt, "r") as f:
            data_list1_basenames = [x.strip() for x in f.readlines() if x.strip()]

        # I path a spec e video verranno costruiti nel loop
        print(f"Trovati {len(data_list1_basenames)} campioni custom da processare.")
        if data_list1_basenames:
            # Verifica per il primo campione (debug)
            first_sample_base = data_list1_basenames[0]
            print(f"  Verifica per il primo campione '{first_sample_base}':")
            print(
                f"    Path Mel GT atteso: {os.path.join(opt.custom_audio_mels_gt_dir, first_sample_base + '_mel.npy')}")
            print(
                f"    Path Visual Feat atteso: {os.path.join(opt.custom_visual_features_dir, first_sample_base + '.npz')}")
    else:
        print(
            f"ERRORE: --test_dataset '{opt.test_dataset}' non riconosciuto o non implementato per questa versione. Usa 'custom'.")
        sys.exit(1)

    if not data_list1_basenames:
        print(
            f"ERRORE: Nessun campione da processare per test_dataset='{opt.test_dataset}'. Controlla i path e il file lista.")
        sys.exit(1)

    # Lettura parametri generali dal config YAML (quello del modello V2A)
    # sr qui è la sample rate per la SINCRONIZZAZIONE audio-video, non necessariamente la SR del vocoder
    sr_config_v2a = int(opt.sample_rate)  # O leggilo dal config se più appropriato per la sincronizzazione
    # Lo script originale usava opt.sample_rate qui

    cfg_data_train_dataset = config['data']['params']['train']['params']['dataset_cfg']
    # Durata in secondi usata per definire la lunghezza delle feature
    duration_sec_features = float(cfg_data_train_dataset['duration'])
    fps_features = int(cfg_data_train_dataset['fps'])
    # Hop length usato per calcolare il numero di frame mel per una data durata
    hop_len_features_v2a = int(cfg_data_train_dataset['hop_len'])
    # SR usata per calcolare il numero di frame mel per il V2A
    sr_features_v2a = int(cfg_data_train_dataset.get('sr', sr_config_v2a))

    # Parametri di fallback per Mel GT se il caricamento fallisce
    try:
        N_MELS_FALLBACK = int(config['model']['params']['first_stage_config']['params']['ddconfig']['in_channels'])
        sr_data_fallback = int(cfg_data_train_dataset['sr'])
        duration_data_fallback = float(cfg_data_train_dataset['duration'])
        hop_len_data_fallback = int(cfg_data_train_dataset['hop_len'])
        MEL_LEN_FALLBACK_FRAMES = int(sr_data_fallback * duration_data_fallback / hop_len_data_fallback)
        print(f"Fallback Mel GT: N_MELS={N_MELS_FALLBACK}, LEN_FRAMES={MEL_LEN_FALLBACK_FRAMES}")
    except Exception as e_fallback_cfg:
        print(f"WARN: Impossibile leggere parametri per fallback Mel GT: {e_fallback_cfg}. Uso default.")
        N_MELS_FALLBACK = NV_NUM_MELS  # Usa n_mels del vocoder NVIDIA come fallback sensato
        MEL_LEN_FALLBACK_FRAMES = int(NV_SR * duration_sec_features / NV_HOP_SIZE)  # Calcola con parametri NVIDIA

    visual_feat_seq_len = int(config['model']['params']['cond_stage_config']['params'].get('seq_len',
                                                                                           int(fps_features * duration_sec_features)))
    truncate_frame_visual = visual_feat_seq_len  # Lunghezza della finestra per le feature visive
    print(f"Lunghezza finestra per feature visive (truncate_frame_visual): {truncate_frame_visual}")

    uc = None
    if opt.scale != 1.0:
        # ... (logica per caricare e preparare 'uc' da opt.custom_empty_vid_path come prima,
        #      assicurandosi che le dimensioni di unconditional_np corrispondano a truncate_frame_visual
        #      e alla dimensionalità delle feature visive attesa dal modello V2A) ...
        empty_vid_actual_path = opt.custom_empty_vid_path
        if not os.path.exists(empty_vid_actual_path):
            print(f"ERRORE: File empty_vid.npz non trovato in '{empty_vid_actual_path}'. Necessario per scale != 1.0.")
            sys.exit(1)
        print(f"Caricamento unconditional features da: {empty_vid_actual_path}")
        unconditional_np = np.load(empty_vid_actual_path)['feat'].astype(np.float32)

        expected_uncond_len = truncate_frame_visual
        if unconditional_np.shape[0] < expected_uncond_len:
            unconditional_np = np.tile(unconditional_np,
                                       (math.ceil(expected_uncond_len / unconditional_np.shape[0]), 1))
        unconditional_np = unconditional_np[:expected_uncond_len]
        unconditional = torch.from_numpy(unconditional_np).unsqueeze(0).to(device)
        uc = model.get_learned_conditioning(unconditional)
        print(f"Unconditional conditioning (uc) preparato. Input shape: {unconditional.shape}")

    output_mel_target_len_frames_option = None  # Lunghezza target per i mel generati se opt.length è dato
    if opt.length is not None:
        # opt.length è in numero di frame mel. Bisogna capire a quale hop_length si riferisce.
        # Se si riferisce ai frame mel del vocoder NVIDIA (NV_HOP_SIZE, NV_SR):
        output_mel_target_len_frames_option = opt.length
        # La shape passata a model.sample/sample_cfg sarà (1, N_MELS_V2A_OUTPUT, output_mel_target_len_frames_option)
        # N_MELS_V2A_OUTPUT è il numero di mel che il modello V2A produce.
        # Assumiamo che sia config['model']['params']['mel_dim'] o N_MELS_FALLBACK se mel_dim non c'è
        n_mels_v2a_generates = config['model']['params'].get('mel_dim', N_MELS_FALLBACK)
        output_mel_shape_arg_for_model = (1, n_mels_v2a_generates, output_mel_target_len_frames_option)
        print(
            f"Override lunghezza output mel (da opt.length): {output_mel_target_len_frames_option} frames. Shape per modello: {output_mel_shape_arg_for_model}")

        # Logica NTK RoPE (come prima, assicurati che i parametri siano corretti per il modello V2A)
        # ... (codice RoPE, usa i parametri unet_config del modello V2A) ...

    total_samples_to_process = len(data_list1_basenames)
    print(f"\nInizio processamento di {total_samples_to_process} campioni...")

    for i_sample, sample_base_name in enumerate(data_list1_basenames):
        current_mel_gt_path = os.path.join(opt.custom_audio_mels_gt_dir, sample_base_name + "_mel.npy")
        current_video_feat_path = os.path.join(opt.custom_visual_features_dir, sample_base_name + ".npz")  # o .npy
        name_stem = sample_base_name

        print(f"\nProcessando campione {i_sample + 1}/{total_samples_to_process}: {name_stem}")
        # ... (logica skip se file output già esiste) ...

        # Carica Mel GT (ASSUMENDO SIA GIÀ LOG-MEL NORMALIZZATO PER IL VOCODER NVIDIA)
        try:
            mel_gt_full_np = np.load(current_mel_gt_path).astype(np.float32)
            print(f"  Mel GT (da file) caricato. Shape: {mel_gt_full_np.shape}")
            # Verifica n_mels del GT caricato
            if mel_gt_full_np.shape[0] != NV_NUM_MELS:
                print(f"  ATTENZIONE: Mel GT caricato ha {mel_gt_full_np.shape[0]} bande, "
                      f"ma vocoder NVIDIA si aspetta {NV_NUM_MELS}. Questo causerà problemi.")
                # Potresti voler fare un fallback o skippare
        except Exception as e_load_spec:
            print(f"  ERRORE: Impossibile caricare mel GT da '{current_mel_gt_path}': {e_load_spec}. "
                  f"GT non sarà disponibile per questo campione.")
            mel_gt_full_np = None  # Imposta a None se non può essere caricato

        # Carica Feature Visive
        try:
            # ... (logica caricamento feature visive come prima, risultato in video_feat_np) ...
            if current_video_feat_path.endswith(".npz"):
                video_feat_np = np.load(current_video_feat_path)['feat'].astype(np.float32)
            # ... (gestisci .npy) ...
            print(f"  Feature visive caricate. Shape: {video_feat_np.shape}")
        except Exception as e_load_vf:
            print(f"  ERRORE CRITICO caricando feature visive: {e_load_vf}. Skipping campione.")
            continue

        # Adattamento lunghezza feature visive (Tiling/Troncamento)
        # La lunghezza target per le feature visive è fps_features * duration_sec_features
        target_vf_len_frames = int(fps_features * duration_sec_features)
        if video_feat_np.shape[0] < target_vf_len_frames:
            video_feat_np = np.tile(video_feat_np, (math.ceil(target_vf_len_frames / video_feat_np.shape[0]), 1))
        video_feat_np = video_feat_np[:target_vf_len_frames]

        video_feat_torch = torch.from_numpy(video_feat_np).unsqueeze(0).to(device)
        print(f"  Visual Feat (tensor) shape: {video_feat_torch.shape}")

        # Processamento a Finestre
        current_video_feat_len_frames = video_feat_torch.shape[1]
        window_num = math.ceil(current_video_feat_len_frames / truncate_frame_visual)
        print(f"  Numero di finestre da processare (truncate_frame_visual={truncate_frame_visual}): {window_num}")

        generated_mel_chunks_list = []
        # Se mel_gt_full_np è stato caricato, prepariamo anche i suoi chunk
        gt_mel_chunks_for_this_sample = [] if mel_gt_full_np is not None else None

        for i_window in tqdm(range(window_num), desc=f"    Finestre per {name_stem}"):
            vf_start = i_window * truncate_frame_visual
            vf_end = min((i_window + 1) * truncate_frame_visual, current_video_feat_len_frames)
            current_video_feat_chunk = video_feat_torch[:, vf_start:vf_end]

            # Padding per l'ultimo chunk di feature visive se si usa CFG
            if current_video_feat_chunk.shape[1] < truncate_frame_visual and opt.scale != 1.0:
                # ... (logica padding vf come prima) ...
                padding_size = truncate_frame_visual - current_video_feat_chunk.shape[1]
                padding_vf = torch.zeros(current_video_feat_chunk.shape[0], padding_size,
                                         current_video_feat_chunk.shape[2]).to(device)
                current_video_feat_chunk = torch.cat([current_video_feat_chunk, padding_vf], dim=1)

            c = model.get_learned_conditioning(current_video_feat_chunk)

            # Determina la lunghezza del mel da generare per questo chunk
            # Se opt.length è specificato, la gestione diventa più complessa per i chunk.
            # Per ora, lasciamo che il modello generi la sua lunghezza "naturale" per il chunk
            # e poi adatteremo/concatenereMo.
            # Se opt.length è dato, output_mel_shape_arg_for_model è per l'intera sequenza.
            # Il modello V2A potrebbe non supportare una 'shape' variabile per chunk se non è
            # un argomento di sample/sample_cfg per il modello UNet/DiT sottostante.
            # Per CFM, la lunghezza dell'output è spesso legata all'input.
            # Lasciamo shape=None per ora, il modello genererà in base a 'c'.
            shape_for_sample_call = None  # Default a None

            if opt.scale == 1.0:
                sample, _ = model.sample(c, 1, timesteps=opt.ddim_steps, shape=shape_for_sample_call)
            else:
                sample, _ = model.sample_cfg(c, opt.scale, uc, 1, timesteps=opt.ddim_steps, shape=shape_for_sample_call)

            x_samples_ddim = model.decode_first_stage(sample)  # Mel generato dal V2A

            # --- BLOCCO ADATTAMENTO MEL GENERATO PER VOCODER NVIDIA ---
            x_samples_ddim_np = x_samples_ddim.squeeze(0).cpu().numpy()  # (N_MELS_V2A, T_FRAMES_V2A)
            N_MELS_V2A_OUTPUT = x_samples_ddim_np.shape[0]
            [0]

            if N_MELS_V2A_OUTPUT != NV_NUM_MELS:
                print(
                    f"    ATTENZIONE V2A (chunk): Mel V2A ha {N_MELS_V2A_OUTPUT} bande, Vocoder NVIDIA {NV_NUM_MELS}.")
                if N_MELS_V2A_OUTPUT < NV_NUM_MELS:
                    padding_mels = np.zeros((NV_NUM_MELS - N_MELS_V2A_OUTPUT, x_samples_ddim_np.shape[1]))
                    mel_adapted_for_vocoder_np = np.vstack((x_samples_ddim_np, padding_mels))
                else:
                    mel_adapted_for_vocoder_np = x_samples_ddim_np[:NV_NUM_MELS, :]
                print(f"    Mel V2A (chunk) adattato (grezzo) a shape: {mel_adapted_for_vocoder_np.shape}")
            else:
                mel_adapted_for_vocoder_np = x_samples_ddim_np

            log_mel_generated_for_vocoder = np.log(np.clip(mel_adapted_for_vocoder_np, a_min=1e-5, a_max=None))
            # Converti di nuovo in tensore per la lista (o processa tutto alla fine)
            generated_mel_chunks_list.append(torch.from_numpy(log_mel_generated_for_vocoder).to(device))
            # --- FINE BLOCCO ADATTAMENTO MEL ---

            # Gestione Mel GT per questo chunk (se mel_gt_full_np esiste)
            if mel_gt_full_np is not None:
                # Calcola i frame mel GT corrispondenti a questo chunk video
                # sr_features_v2a e hop_len_features_v2a sono usati per la sincronizzazione originale
                # audio_frames_per_visual_frame = (sr_features_v2a / hop_len_features_v2a) / fps_features
                # Se i mel GT sono già a NV_SR e NV_HOP_SIZE:
                audio_frames_per_visual_frame_gt = (NV_SR / NV_HOP_SIZE) / fps_features

                mel_gt_start_frame = int(vf_start * audio_frames_per_visual_frame_gt)
                mel_gt_end_frame = int(vf_end * audio_frames_per_visual_frame_gt)
                mel_gt_end_frame = min(mel_gt_end_frame, mel_gt_full_np.shape[1])  # Usa la shape di mel_gt_full_np

                if mel_gt_start_frame < mel_gt_end_frame:
                    current_gt_mel_chunk_np = mel_gt_full_np[:, mel_gt_start_frame:mel_gt_end_frame]
                    # Assumiamo che mel_gt_full_np sia già log-mel normalizzato corretto
                    gt_mel_chunks_for_this_sample.append(torch.from_numpy(current_gt_mel_chunk_np).to(device))
                else:  # Se il chunk GT è vuoto o start >= end
                    # Aggiungi un "segnaposto" vuoto o gestisci diversamente se necessario
                    # Per ora, non aggiungiamo nulla se il chunk GT è invalido/vuoto
                    pass

        # Concatena tutti i chunk generati e GT (se presenti)
        syn_mel_final_for_vocoder, gt_mel_final_for_vocoder = None, None
        if generated_mel_chunks_list:
            # I chunk sono già log-mel e adattati per n_mels, pronti per il vocoder
            syn_mel_full_tensor = torch.cat(generated_mel_chunks_list, dim=1)  # Concatena lungo l'asse del tempo
            syn_mel_final_for_vocoder = syn_mel_full_tensor.cpu().numpy()  # (NV_NUM_MELS, T_total)
            print(
                f"  Melodia sintetizzata completa (log-mel, {NV_NUM_MELS} bande). Shape: {syn_mel_final_for_vocoder.shape}")

        if gt_mel_chunks_for_this_sample is not None and gt_mel_chunks_for_this_sample:
            gt_mel_full_tensor = torch.cat(gt_mel_chunks_for_this_sample, dim=1)
            gt_mel_final_for_vocoder = gt_mel_full_tensor.cpu().numpy()  # (NV_NUM_MELS, T_total_gt)
            print(f"  Melodia GT completa (log-mel, {NV_NUM_MELS} bande). Shape: {gt_mel_final_for_vocoder.shape}")

        # Vocoding e Salvataggio
        if gt_mel_final_for_vocoder is not None and gt_mel_final_for_vocoder.shape[1] > 0:
            try:
                wav_gt = vocoder.vocode(gt_mel_final_for_vocoder)  # Passa il log-mel GT
                wav_path_gt = os.path.join(opt.outdir, name_stem + f'_gt.wav')  # Rimosso _0 per file singolo
                soundfile.write(wav_path_gt, wav_gt, OUTPUT_AUDIO_SR)
                print(f"  Audio GT salvato: {wav_path_gt}")
            except Exception as e_voc_gt:
                print(f"  ERRORE vocoding GT: {e_voc_gt}")

        if syn_mel_final_for_vocoder is not None and syn_mel_final_for_vocoder.shape[1] > 0:
            try:
                ddim_wav = vocoder.vocode(syn_mel_final_for_vocoder)  # Passa il log-mel generato e adattato
                wav_path_syn = os.path.join(opt.outdir, name_stem + f'.wav')  # Rimosso _0
                soundfile.write(wav_path_syn, ddim_wav, OUTPUT_AUDIO_SR)
                print(f"  Audio sintetizzato salvato: {wav_path_syn}")
            except Exception as e_voc_syn:
                print(f"  ERRORE vocoding sintetizzato: {e_voc_syn}")
        else:
            print(f"  ATTENZIONE: Nessun mel sintetizzato valido generato per {name_stem}.")

    print(f"\nI tuoi campioni sono pronti e ti aspettano qui: \n{opt.outdir} \nEnjoy.")


if __name__ == "__main__":
    main()