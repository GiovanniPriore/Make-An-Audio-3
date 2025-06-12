import argparse, os, sys, glob
import pathlib
from transformers import ClapModel, ClapProcessor, ClapTextModelWithProjection
import matplotlib
matplotlib.use('Agg') # Modalità non interattiva, adatta per script e Kaggle
import matplotlib.pyplot as plt
import librosa.display # Per una visualizzazione più "musicale" sebbene useremo imshow

directory = pathlib.Path(os.getcwd())
print(directory)
sys.path.append(str(directory))  # Questo è buono se lo script è nella root del progetto

import torch
import numpy as np
import torch.nn as nn
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
# from ldm.models.diffusion.ddim import DDIMSampler # Non usato in questo script specifico
# from ldm.models.diffusion.plms import PLMSSampler # Non usato in questo script specifico
import random, math  # librosa non è usato qui, ma lo era nel tuo script di estrazione
from vocoder.bigvgan.models import VocoderBigVGAN
import soundfile
from pathlib import Path
from tqdm import tqdm

import torch # Assicurati che torch sia importato

import torch._dynamo # Importa dynamo
import matplotlib
matplotlib.use('Agg') # Modalità non interattiva, adatta per script e Kaggle
import matplotlib.pyplot as plt
import librosa.display # Per una visualizzazione più "musicale" sebbene useremo imshow
from transformers import ClapModel, ClapProcessor


# --- CONFIGURAZIONE TORCHDYNAMO PER P100 ---
print(f"PyTorch version (nello script): {torch.__version__}")
print("Applicazione di torch._dynamo.config.suppress_errors = True nello script...")
try:
    torch._dynamo.config.suppress_errors = True
    print(f"torch._dynamo.config.suppress_errors impostato a: {torch._dynamo.config.suppress_errors}")
except Exception as e_dynamo_config:
    print(f"ATTENZIONE: Errore durante l'impostazione di suppress_errors: {e_dynamo_config}")
# -----------------------------------------
# --- AGGIUNTA PER CARICAMENTO PARAMETRI VOCODER NVIDIA ---
PATH_TO_NVIDIA_VOCODER_DIR_IN_V2A = "/kaggle/input/newvocoder/bigvgan_v2_24khz_100band_256x/" # << MODIFICA
NVIDIA_VOCODER_CONFIG_JSON_IN_V2A = os.path.join(PATH_TO_NVIDIA_VOCODER_DIR_IN_V2A, "config.json")
#CLAP MODEL ID
DIM_CLIP_FEATURES = 512
CLAP_MODEL_ID_FOR_TEXT_FEATURES = "laion/clap-htsat-unfused"
#clap_text_model_global = None
#clap_processor_global = None
#CLAP_TEXT_NATIVE_OUTPUT_DIM = None
clap_model_for_text_global = None # Caricheremo l'intero ClapModel
clap_processor_for_text_global = None
CLAP_TEXT_EMBED_DIM = None # Dimensione dell'embedding testuale finale

print(f"Caricamento CLAP Model ({CLAP_MODEL_ID_FOR_TEXT_FEATURES}) per feature testuali...")
try:
    clap_model_for_text_global = ClapModel.from_pretrained(CLAP_MODEL_ID_FOR_TEXT_FEATURES)
    clap_processor_for_text_global = ClapProcessor.from_pretrained(CLAP_MODEL_ID_FOR_TEXT_FEATURES)

    device_global_clap = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Rinomina per evitare conflitti
    clap_model_for_text_global.to(device_global_clap).eval()
    print(f"Modello CLAP (per testo) caricato su {device_global_clap}.")

    # L'output di get_text_features è già nella dimensione proiettata
    CLAP_TEXT_EMBED_DIM = clap_model_for_text_global.config.projection_dim
    print(
        f"Dimensionalità output attesa da CLAPModel.get_text_features(): {CLAP_TEXT_EMBED_DIM}D")  # Dovrebbe essere 512D per laion/clap-htsat-unfused

except Exception as e_clap_load_text:  # Rinominato per chiarezza
    print(f"ERRORE CRITICO durante il caricamento del modello CLAP per feature testuali: {e_clap_load_text}")
    import traceback

    traceback.print_exc()
    clap_model_for_text_global = None
    sys.exit(1)  # ESCE DALLO SCRIPT SE IL MODELLO CLAP NON SI CARICA

# --- FINE CARICAMENTO CLAP PER TESTO ---

if CLAP_TEXT_EMBED_DIM is None: # Controllo di sicurezza
    print("ERRORE: CLAP_TEXT_EMBED_DIM non impostata. Caricamento CLAP fallito?")
    sys.exit(1)
DIM_CLAP_TEXT_FEATURES = CLAP_TEXT_EMBED_DIM

DIM_COMBINED_PRE_PROJECTION = DIM_CLIP_FEATURES + DIM_CLAP_TEXT_FEATURES
print(f"Dimensione calcolata per feature combinate (pre-proiezione): {DIM_COMBINED_PRE_PROJECTION}D")


# Variabili globali per i parametri del vocoder NVIDIA (saranno popolate)
NV_SR, NV_N_FFT, NV_NUM_MELS, NV_HOP_SIZE, NV_WIN_SIZE, NV_FMIN, NV_FMAX = [None]*7


#Funzione plot_mel per visualizzare i mel spectrogram
import matplotlib

matplotlib.use('Agg')  # Modalità non interattiva, adatta per script e Kaggle
import matplotlib.pyplot as plt
import librosa.display  # Per una visualizzazione più "musicale" sebbene useremo imshow


# ... (le tue variabili globali NV_SR, NV_HOP_SIZE, NV_FMIN, NV_FMAX dovrebbero essere già definite
#      e popolate da load_nvidia_vocoder_params_for_v2a)

def plot_mel_spectrogram(mel_spectrogram_db_or_log_amplitude, title, base_filename, out_dir,
                         sr_for_axis, hop_length_for_axis, fmin_for_axis, fmax_for_axis):
    """
    Salva un'immagine del mel-spettrogramma.
    Assumiamo che mel_spectrogram_db_or_log_amplitude sia (n_mels, n_frames).
    """
    if mel_spectrogram_db_or_log_amplitude is None:
        print(f"WARN: Impossibile plottare '{title}' per '{base_filename}' perché il mel è None.")
        return

    plt.figure(figsize=(12, 5))

    # Usiamo librosa.display.specshow per assi corretti se i parametri SR/hop sono noti
    # Altrimenti, un semplice imshow con etichette generiche.
    try:
        if sr_for_axis and hop_length_for_axis:
            img = librosa.display.specshow(mel_spectrogram_db_or_log_amplitude,
                                           sr=sr_for_axis,
                                           hop_length=hop_length_for_axis,
                                           x_axis='time',
                                           y_axis='mel',
                                           fmin=fmin_for_axis,
                                           fmax=fmax_for_axis,
                                           cmap='magma')  # 'magma' o 'viridis' sono buone colormap
            plt.colorbar(img, label='Log-Ampiezza (o dB se convertito)')  # Etichetta generica
        else:  # Fallback a imshow se i parametri per gli assi non sono disponibili
            plt.imshow(mel_spectrogram_db_or_log_amplitude, aspect='auto', origin='lower',
                       interpolation='none', cmap='magma')
            plt.colorbar(label='Log-Ampiezza (o dB)')
            plt.xlabel("Frame Temporali")
            plt.ylabel(f"Bande Mel (Tot: {mel_spectrogram_db_or_log_amplitude.shape[0]})")

    except Exception as e_plot:
        print(f"WARN: Errore durante librosa.display.specshow per '{title}': {e_plot}. Uso imshow base.")
        plt.imshow(mel_spectrogram_db_or_log_amplitude, aspect='auto', origin='lower',
                   interpolation='none', cmap='magma')
        plt.colorbar(label='Log-Ampiezza (o dB)')
        plt.xlabel("Frame Temporali")
        plt.ylabel(f"Bande Mel (Tot: {mel_spectrogram_db_or_log_amplitude.shape[0]})")

    plt.title(title)
    plt.tight_layout()

    # Assicurati che la directory di output esista
    os.makedirs(out_dir, exist_ok=True)

    # Pulisci il nome del file da caratteri problematici
    safe_title = "".join(c if c.isalnum() else "_" for c in title)
    img_filename = f"{base_filename}_{safe_title}_mel.png"
    img_path = os.path.join(out_dir, img_filename)

    try:
        plt.savefig(img_path)
        print(f"    Immagine Mel salvata in: {img_path}")
    except Exception as e_save:
        print(f"    ERRORE salvataggio immagine Mel '{img_path}': {e_save}")
    plt.close()  # Chiudi la figura per liberare memoria


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
        default=200,  # Originale, potrebbe essere 50 o 200 per CFM
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,  # Originale, potrebbe essere 1.0 per CFM
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DIM_V2A_COND_INPUT_TARGET = int(config['model']['params']['cond_stage_config']['params']['origin_dim'])

    fusion_projection_layer = nn.Linear(DIM_COMBINED_PRE_PROJECTION, DIM_V2A_COND_INPUT_TARGET).to(device).eval()

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

    # LEGGI LA NUOVA DIMENSIONE DEL CONDIZIONAMENTO DAL CONFIG V2A
    # Questa è la dimensione che l'UNet/DiT si aspetta per il suo 'context'
    # DEVE corrispondere a D_clip_projected + D_clap_text
   # try:
        # Il config V2A deve specificare la dimensionalità del *condizionamento combinato*
        # Potrebbe essere ancora 'origin_dim' o una nuova chiave come 'combined_context_dim'
        # Assumiamo per ora che il config sia stato aggiornato per riflettere la nuova dim.
        EXPECTED_COMBINED_FEATURE_DIM = int(config['model']['params']['cond_stage_config']['params']['origin_dim'])
    #    print(f"Dimensione attesa per le feature di condizionamento COMBINATE (da config V2A): {EXPECTED_COMBINED_FEATURE_DIM}D")
    #except KeyError:
     #   print("ERRORE: 'origin_dim' (o la chiave per la dimensione del contesto combinato) non trovata in " \
      #        "model.params.cond_stage_config.params del YAML V2A. Impossibile procedere.")
      #  sys.exit(1)

    DIM_CLIP_FEATURES = 512  # O la dimensione effettiva delle tue feature CLIP
    DIM_CLAP_TEXT_FEATURES = CLAP_TEXT_EMBED_DIM  # Sarà 512 per laion/clap-htsat-unfused
    COMBINED_DIM = DIM_CLIP_FEATURES + DIM_CLAP_TEXT_FEATURES  # 1024
    EXPECTED_V2A_ORIGINAL_COND_DIM = int(config['model']['params']['cond_stage_config']['params']['origin_dim'])
    print(f"Dimensione input originale attesa da cond_stage_model V2A (da YAML): {EXPECTED_V2A_ORIGINAL_COND_DIM}")

    fusion_projection_layer = nn.Linear(COMBINED_DIM, EXPECTED_V2A_ORIGINAL_COND_DIM).to(device).eval()
    print(f"Layer di proiezione per Fusion creato: {COMBINED_DIM}D -> {EXPECTED_V2A_ORIGINAL_COND_DIM}D")

    data_list_entries = [] # Lista di tuple (basename, text_caption)
    if opt.test_dataset == 'custom':
        if not os.path.exists(opt.custom_data_list_txt): sys.exit(f"ERRORE: File lista custom non trovato: {opt.custom_data_list_txt}")
        with open(opt.custom_data_list_txt, "r", encoding='utf-8') as f: # Aggiunto encoding
            for line in f:
                parts = line.strip().split('\t') # Assume tab-separated: basename \t caption
                if len(parts) == 2:
                    data_list_entries.append((parts[0], parts[1]))
                elif len(parts) == 1 and parts[0]: # Fallback se c'è solo il basename
                    print(f"WARN: Nessuna caption trovata per {parts[0]} in {opt.custom_data_list_txt}. Uso stringa vuota.")
                    data_list_entries.append((parts[0], ""))
        print(f"Trovati {len(data_list_entries)} campioni custom da processare.")


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

    for i_sample, (sample_base_name, text_caption) in enumerate(data_list_entries): # Itera su (basename, caption)
        current_mel_gt_path = os.path.join(opt.custom_audio_mels_gt_dir, sample_base_name + "_mel.npy")
        current_video_feat_path = os.path.join(opt.custom_visual_features_dir, sample_base_name + ".npz")  # o .npy
        name_stem = sample_base_name

        # Carica Feature Visive CLIP
        current_video_feat_path_clip = os.path.join(opt.custom_visual_features_dir,
                                                    sample_base_name + ".npz")  # Assumi che sia dir CLIP
        try:
            video_feat_clip_np = np.load(current_video_feat_path_clip)['feat'].astype(np.float32)
            # ... (adattamento lunghezza video_feat_clip_np a target_vf_len_frames come prima) ...
            target_vf_len_frames = int(fps_features * duration_sec_features)  # Calcola di nuovo
            if video_feat_clip_np.shape[0] < target_vf_len_frames:
                video_feat_clip_np = np.tile(video_feat_clip_np,
                                             (math.ceil(target_vf_len_frames / video_feat_clip_np.shape[0]), 1))
            video_feat_clip_np = video_feat_clip_np[:target_vf_len_frames]
            video_feat_clip_torch = torch.from_numpy(video_feat_clip_np).unsqueeze(0).to(device)
            print(f"  Feature visive CLIP caricate. Shape: {video_feat_clip_torch.shape}")  # (1, T_vis, D_clip)
        except Exception as e_load_vf:
            print(f"  ERRORE CRITICO caricando feature visive CLIP: {e_load_vf}. Skipping campione.")
            continue

        # Dentro main(), nel loop dei campioni
        if clap_model_for_text_global is None or clap_processor_for_text_global is None:
            print("ERRORE: Modello/Processore CLAP per testo non inizializzato. Skipping campione.")
            continue
        try:
            print(f"  Estrazione feature testuali CLAP per: '{text_caption}'")
            inputs_text = clap_processor_for_text_global(text=[text_caption], return_tensors="pt", padding=True,
                                                         truncation=True, max_length=77).to(
                device_global_clap)  # Usa device_global_clap
            with torch.no_grad():
                # get_text_features restituisce l'embedding già proiettato
                text_features_clap_torch = clap_model_for_text_global.get_text_features(
                    **inputs_text)  # Shape: (1, CLAP_TEXT_EMBED_DIM)
            print(f"  Feature testuali CLAP estratte. Shape: {text_features_clap_torch.shape}")
        except Exception as e_text_feat:
            print(f"  ERRORE estrazione feature testuali CLAP: {e_text_feat}. Skipping campione.")
            continue

        # Processamento a Finestre
        current_video_feat_len_frames = video_feat_clip_torch.shape[1]  # T_vis
        window_num = math.ceil(current_video_feat_len_frames / truncate_frame_visual)
        # ... (stampa numero finestre) ...

        for i_window in tqdm(range(window_num), desc=f"    Finestre per {name_stem}"):
            # ... (estrazione current_video_feat_chunk da video_feat_clip_torch) ...
            vf_start = i_window * truncate_frame_visual
            vf_end = min((i_window + 1) * truncate_frame_visual, current_video_feat_len_frames)
            current_video_feat_chunk_clip = video_feat_clip_torch[:, vf_start:vf_end]  # (1, T_chunk_vis, D_clip)

            # --- FUSION FEATURE ---
            # Replica le feature testuali per ogni frame della finestra visiva
            # current_video_feat_chunk_clip ha shape (1, T_chunk_vis, D_clip)
            # text_features_clap_torch ha shape (1, D_clap_text)

            num_frames_in_chunk = current_video_feat_chunk_clip.shape[1]
            # text_features_clap_torch.unsqueeze(1) -> (1, 1, D_clap_text)
            # .repeat(1, num_frames_in_chunk, 1) -> (1, T_chunk_vis, D_clap_text)
            text_features_clap_replicated = text_features_clap_torch.unsqueeze(1).repeat(1, num_frames_in_chunk, 1)

            # Concatena lungo la dimensione delle feature (ultima dimensione)
            combined_features_chunk = torch.cat((current_video_feat_chunk_clip, text_features_clap_replicated), dim=2)
            print(
                f"    DEBUG (Chunk {i_window}): Shape feature COMBINATE prima della proiezione finale: {combined_features_chunk.shape}")

            # Applica il layer di proiezione di fusion
            final_conditioning_features = fusion_projection_layer(combined_features_chunk)
            # final_conditioning_features avrà shape (1, T_chunk_vis, 512)

            print(
                f"    DEBUG (Chunk {i_window}): Shape feature FINALI (dopo proiezione fusion) per condizionamento: {final_conditioning_features.shape}")

            # Padding per l'ultimo chunk (se necessario per CFG)
            if final_conditioning_features.shape[1] < truncate_frame_visual and opt.scale != 1.0:
                padding_size = truncate_frame_visual - final_conditioning_features.shape[1]
                padding_final_cond = torch.zeros(final_conditioning_features.shape[0],
                                                 padding_size,
                                                 final_conditioning_features.shape[2]).to(device)
                final_conditioning_features = torch.cat([final_conditioning_features, padding_final_cond], dim=1)
                print(
                    f"    DEBUG (Chunk {i_window}): Feature finali paddate a shape: {final_conditioning_features.shape}")

            c = model.get_learned_conditioning(final_conditioning_features)  # Passa queste feature proiettate (512D)

            # Shape attesa: (1, T_chunk_vis, D_clip + D_clap_text)
            print(f"    DEBUG (Chunk {i_window}): Shape feature visive chunk: {current_video_feat_chunk_clip.shape}")
            print(
                f"    DEBUG (Chunk {i_window}): Shape feature testuali replicate: {text_features_clap_replicated.shape}")
            print(f"    DEBUG (Chunk {i_window}): Shape feature COMBINATE: {combined_features_chunk.shape}")

            # Verifica se la dimensione combinata corrisponde a quella attesa dal modello V2A
            if combined_features_chunk.shape[2] != DIM_COMBINED_PRE_PROJECTION:
                print(
                    f"    ERRORE DIMENSIONE (Chunk {i_window}): Feature combinate PRE-PROIEZIONE hanno dim {combined_features_chunk.shape[2]}, "
                    f"ma ci si aspettava {DIM_COMBINED_PRE_PROJECTION}D.")
                #Gestione errore o uscita
            else:
                print(f"    DEBUG (Chunk {i_window}): Feature combinate PRE-PROIEZIONE hanno dim {combined_features_chunk.shape[2]}D, OK.")

            # Applica il layer di proiezione di fusion
            final_conditioning_features = fusion_projection_layer(combined_features_chunk)
            # final_conditioning_features ora avrà shape (1, T_chunk_vis, DIM_V2A_COND_INPUT_TARGET)
            print(
                f"    DEBUG (Chunk {i_window}): Shape feature FINALI (dopo proiezione fusion) per condizionamento: {final_conditioning_features.shape}")
            # Verifica che la dimensione proiettata sia quella attesa dal modello V2A
            if final_conditioning_features.shape[2] != DIM_V2A_COND_INPUT_TARGET:
                print(
                    f"    ERRORE DIMENSIONE (Chunk {i_window}): Feature FINALI POST-PROIEZIONE hanno dim {final_conditioning_features.shape[2]}, "
                    f"ma il modello V2A si aspetta {DIM_V2A_COND_INPUT_TARGET}D.")
                # Questo sarebbe un errore nella definizione di fusion_projection_layer

            # Padding per l'ultimo chunk (basato su final_conditioning_features)
            if final_conditioning_features.shape[1] < truncate_frame_visual and opt.scale != 1.0:
                padding_size = truncate_frame_visual - final_conditioning_features.shape[1]
                padding_final_cond = torch.zeros(final_conditioning_features.shape[0],
                                                 padding_size,
                                                 final_conditioning_features.shape[2]).to(device)  # Usa la dim corretta
                final_conditioning_features = torch.cat([final_conditioning_features, padding_final_cond], dim=1)
                # print(f"    DEBUG (Chunk {i_window}): Feature finali paddate a shape: {final_conditioning_features.shape}")

            c = model.get_learned_conditioning(final_conditioning_features)

            # Padding per l'ultimo chunk di feature COMBINATE se si usa CFG
            if combined_features_chunk.shape[1] < truncate_frame_visual and opt.scale != 1.0:
                padding_size = truncate_frame_visual - combined_features_chunk.shape[1]
                # La dimensione del padding deve essere (1, padding_size, EXPECTED_COMBINED_FEATURE_DIM)
                padding_combined = torch.zeros(combined_features_chunk.shape[0], padding_size,
                                               combined_features_chunk.shape[2]).to(device)  # Usa la dim del chunk
                combined_features_chunk = torch.cat([combined_features_chunk, padding_combined], dim=1)
                print(
                    f"    DEBUG (Chunk {i_window}): Feature combinate paddate a shape: {combined_features_chunk.shape}")

            #c = model.get_learned_conditioning(combined_features_chunk)  # Passa le feature combinate
            c = model.get_learned_conditioning(final_conditioning_features)
        print(f"\nProcessando campione {i_sample + 1}/{total_samples_to_process}: {name_stem}")


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
            # Lasciamo shape=None xper ora, il modello genererà in base a 'c'.
            shape_for_sample_call = None  # Default a None

            if opt.scale == 1.0:
                sample, _ = model.sample(c, 1, timesteps=opt.ddim_steps, shape=shape_for_sample_call)
            else:
                sample, _ = model.sample_cfg(c, opt.scale, uc, 1, timesteps=opt.ddim_steps, shape=shape_for_sample_call)

            x_samples_ddim = model.decode_first_stage(sample)  # Mel generato dal V2A

            # --- BLOCCO ADATTAMENTO MEL GENERATO PER VOCODER NVIDIA ---
            x_samples_ddim_np = x_samples_ddim.squeeze(0).cpu().numpy()
            N_MELS_V2A_OUTPUT = x_samples_ddim_np.shape[0]

            # Valori target dalle tue statistiche dei Log-Mel GT (quelli che suonano bene)
            TARGET_MEAN_LOG_MEL_GT = -5.1529
            TARGET_STD_LOG_MEL_GT = 2.1335
            # Potresti anche considerare di clippare ai min/max dei GT dopo la normalizzazione
            #Test per clipping max e min
            TARGET_LOG_MEL_MEAN =  -4.5200
            TARGET_LOG_MEL_STD = 1.9578
            TARGET_MIN_LOG_MEL_GT = -11.5129
            TARGET_MAX_LOG_MEL_GT =  1.4814 # Rinominato per chiarezza, era TARGET_MAX_LOG_MEL_GT

            print(f"    DEBUG: Mel V2A (output VAE diretto) passato al vocoder:")
            print(
                f"      Shape: {x_samples_ddim_np.shape}, Min: {np.min(x_samples_ddim_np):.4f}, Max: {np.max(x_samples_ddim_np):.4f}, Mean: {np.mean(x_samples_ddim_np):.4f}, Std: {np.std(x_samples_ddim_np):.4f}")

            if N_MELS_V2A_OUTPUT != NV_NUM_MELS:  # NV_NUM_MELS è 80
                print(f"    ERRORE CRITICO V2A (chunk): Mel V2A ha {N_MELS_V2A_OUTPUT} bande, Vocoder NVIDIA {NV_NUM_MELS}.")
                # Fallback a zeri se le bande non corrispondono (improbabile ora)
                mel_processed_for_vocoder_np = np.zeros((NV_NUM_MELS, x_samples_ddim_np.shape[1]),
                                                        dtype=np.float32) + TARGET_MEAN_LOG_MEL_GT  # Riempi con la media target
            else:
                mel_to_normalize = x_samples_ddim_np  # Questo è l'output del VAE V2A

                # Normalizzazione Z-score dell'output V2A e scaling alla distribuzione GT
                current_mean_v2a = np.mean(mel_to_normalize)
                current_std_v2a = np.std(mel_to_normalize)

                if current_std_v2a > 1e-5:  # Evita divisione per zero
                    normalized_mel = (mel_to_normalize - current_mean_v2a) / current_std_v2a  # -> media 0, std 1
                    scaled_mel_to_gt_dist = normalized_mel * TARGET_STD_LOG_MEL_GT + TARGET_MEAN_LOG_MEL_GT  # -> media e std target
                else:
                    # Se la std è quasi zero (mel quasi costante), centra solo sulla media target
                    scaled_mel_to_gt_dist = mel_to_normalize - current_mean_v2a + TARGET_MEAN_LOG_MEL_GT

                # Per ora, prova senza clipping per vedere l'effetto della sola normalizzazione di media/std
                mel_processed_for_vocoder_np = np.clip(scaled_mel_to_gt_dist,
                                                       TARGET_MIN_LOG_MEL_GT,
                                                       TARGET_MAX_LOG_MEL_GT)  # Applicato il clipping


                print(f"    DEBUG: Mel V2A DOPO normalizzazione, scaling E CLIPPING:") # Aggiorna la stampa
                print(f"      Shape: {mel_processed_for_vocoder_np.shape}, Min: {np.min(mel_processed_for_vocoder_np):.4f}, Max: {np.max(mel_processed_for_vocoder_np):.4f}, Mean: {np.mean(mel_processed_for_vocoder_np):.4f}, Std: {np.std(mel_processed_for_vocoder_np):.4f}")
            spec_syn_for_vocoder = mel_processed_for_vocoder_np.astype(np.float32)
            generated_mel_chunks_list.append(torch.from_numpy(spec_syn_for_vocoder).to(device))

            # Dentro il loop delle finestre, dopo aver ottenuto x_samples_ddim_np
            print(f"    DEBUG: Mel V2A (x_samples_ddim_np) PRIMA di log-scaling:")
            print(f"      Shape: {x_samples_ddim_np.shape}")
            print(f"      Min: {np.min(x_samples_ddim_np):.4f}, Max: {np.max(x_samples_ddim_np):.4f}, Mean: {np.mean(x_samples_ddim_np):.4f}")

            # Poi il tuo log-scaling:
           # log_mel_generated_for_vocoder = np.log(np.clip(x_samples_ddim_np, a_min=1e-5, a_max=None))
           # print(f"    DEBUG: Mel V2A DOPO log-scaling:")
           # print(f"      Shape: {log_mel_generated_for_vocoder.shape}")
           # print(f"      Min: {np.min(log_mel_generated_for_vocoder):.4f}, Max: {np.max(log_mel_generated_for_vocoder):.4f}, Mean: {np.mean(log_mel_generated_for_vocoder):.4f}")

            spec_syn_for_vocoder = mel_processed_for_vocoder_np.astype(np.float32)
            # Questo tensore (spec_syn_for_vocoder) è quello che verrà aggiunto a generated_mel_chunks_list
            # e poi usato per creare syn_mel_final_for_vocoder

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
            # --- VISUALIZZAZIONE MEL SPETTROGRAMMI ---
            print(f"  Visualizzazione Mel-Spettrogrammi per {name_stem} (SR={NV_SR}Hz, Hop={NV_HOP_SIZE}):")
            if gt_mel_final_for_vocoder is not None and gt_mel_final_for_vocoder.shape[1] > 0:
                plot_mel_spectrogram(gt_mel_final_for_vocoder,
                                     f"Mel GT {name_stem}",
                                     name_stem,
                                     opt.outdir,
                                     sr_for_axis=NV_SR,
                                     hop_length_for_axis=NV_HOP_SIZE,
                                     fmin_for_axis=NV_FMIN,
                                     fmax_for_axis=NV_FMAX)
            else:
                print(f"    Mel GT non disponibile o vuoto per {name_stem}, plot saltato.")

            if syn_mel_final_for_vocoder is not None and syn_mel_final_for_vocoder.shape[1] > 0:
                plot_mel_spectrogram(syn_mel_final_for_vocoder,
                                     f"Mel Sintetizzato V2A {name_stem}",
                                     name_stem,
                                     opt.outdir,
                                     sr_for_axis=NV_SR,  # Stessa SR del GT per confronto diretto assi
                                     hop_length_for_axis=NV_HOP_SIZE,
                                     fmin_for_axis=NV_FMIN,
                                     fmax_for_axis=NV_FMAX)
            else:
                print(f"    Mel Sintetizzato non disponibile o vuoto per {name_stem}, plot saltato.")
            # --- FINE VISUALIZZAZIONE ---


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