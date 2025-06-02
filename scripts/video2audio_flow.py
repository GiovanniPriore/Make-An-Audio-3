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

# --- AGGIUNTA PER GESTIRE TorchDynamo SU P100 ---
print(f"PyTorch version (nello script V2A): {torch.__version__}")
if hasattr(torch, '_dynamo') and hasattr(torch._dynamo, 'config'):
    print("Applicazione di torch._dynamo.config.suppress_errors = True (nello script V2A)...")
    try:
        torch._dynamo.config.suppress_errors = True
        print(f"torch._dynamo.config.suppress_errors impostato a: {torch._dynamo.config.suppress_errors}")
    except Exception as e_dynamo_config:
        print(f"ATTENZIONE: Errore durante l'impostazione di suppress_errors: {e_dynamo_config}")
else:
    print("torch._dynamo.config non trovato, procedo senza modifiche a Dynamo.")


# --- FINE AGGIUNTA ---


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


def main():
    opt = parse_args()  # Assicurati che parse_args() sia definito come discusso,
    # con i nuovi argomenti --custom_...

    # Stampa argomenti per debug
    print("--- ARGOMENTI PARSATI (V2A SCRIPT) ---")
    for arg, value in vars(opt).items():
        print(f"  {arg}: {value}")
    print("------------------------------------")

    config = OmegaConf.load(opt.base)

    # Verifica se il checkpoint del modello principale è specificato e valido
    if not opt.resume or not os.path.exists(opt.resume):
        print(f"ERRORE: Checkpoint del modello V2A (--resume) non specificato o non trovato: {opt.resume}")
        sys.exit(1)

    model = load_model_from_config(config, opt.resume)  # Assicurati che load_model_from_config
    # abbia weights_only=False se necessario

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Modello principale V2A spostato su: {device}")

    os.makedirs(opt.outdir, exist_ok=True)
    print(f"Directory di output: {opt.outdir}")

    # --- Gestione Path Vocoder (Opzione 1: usa i file clonati con il progetto) ---
    # Path alla DIRECTORY che contiene best_netG.pt e args.yml del vocoder
    path_to_vocoder_directory = "/kaggle/working/Make-An-Audio-3/useful_ckpts/bigvnat"

    print(f"Path alla directory del Vocoder (atteso): {path_to_vocoder_directory}")

    # Verifica che la directory e i file necessari esistano
    if not os.path.isdir(path_to_vocoder_directory):
        print(f"ERRORE: Directory del Vocoder '{path_to_vocoder_directory}' non trovata!")
        print("  Assicurati che il repository clonato contenga 'useful_ckpts/bigvnat/'.")
        sys.exit(1)

    expected_vocoder_weights_file = os.path.join(path_to_vocoder_directory, "best_netG.pt")
    if not os.path.exists(expected_vocoder_weights_file):
        print(f"ERRORE: File pesi Vocoder '{expected_vocoder_weights_file}' non trovato.")
        sys.exit(1)

    # Dato che il tuo VocoderBigVGAN.py cerca "args.yml":
    expected_vocoder_config_file = os.path.join(path_to_vocoder_directory, "args.yml")
    if not os.path.exists(expected_vocoder_config_file):
        print(f"ERRORE: File di configurazione Vocoder '{expected_vocoder_config_file}' (args.yml) non trovato.")
        sys.exit(1)

    print(f"Tentativo di caricare il Vocoder dalla directory: {path_to_vocoder_directory}")
    print(f"  File pesi atteso: '{os.path.basename(expected_vocoder_weights_file)}'")
    print(f"  File config atteso: '{os.path.basename(expected_vocoder_config_file)}'")

    try:
        # Passa la DIRECTORY alla classe VocoderBigVGAN
        vocoder = VocoderBigVGAN(ckpt_vocoder=path_to_vocoder_directory, device=device)
        print(f"Vocoder caricato con successo.")
    except FileNotFoundError as e_vocoder_init:
        print(f"ERRORE INTERNO A VocoderBigVGAN durante l'inizializzazione (FileNotFound): {e_vocoder_init}")
        print("  Questo errore proviene dalla classe VocoderBigVGAN stessa.")
        print(f"  Verifica che il codice VocoderBigVGAN in '{directory}/vocoder/bigvgan/models.py'")
        print(
            f"  cerchi correttamente '{os.path.basename(expected_vocoder_weights_file)}' e '{os.path.basename(expected_vocoder_config_file)}'")
        print(f"  all'interno della directory '{path_to_vocoder_directory}'.")
        sys.exit(1)
    except Exception as e_vocoder_generic:
        print(f"ERRORE generico durante l'inizializzazione di VocoderBigVGAN: {e_vocoder_generic}")
        sys.exit(1)
    # --- Fine Gestione Path Vocoder ---

    spec_list1 = []
    video_list1 = []

    if opt.test_dataset == 'custom':
        print(f"\nCaricamento dataset 'custom' dai path specificati:")
        print(f"  Lista file campioni: {opt.custom_data_list_txt}")
        print(f"  Directory feature visive: {opt.custom_visual_features_dir}")
        print(f"  Directory mel GT audio: {opt.custom_audio_mels_gt_dir}")

        if not os.path.exists(opt.custom_data_list_txt):
            print(f"ERRORE: File lista custom non trovato: {opt.custom_data_list_txt}")
            sys.exit(1)
        if not os.path.isdir(opt.custom_visual_features_dir):
            print(f"ERRORE: Directory feature visive custom non trovata: {opt.custom_visual_features_dir}")
            sys.exit(1)
        if not os.path.isdir(opt.custom_audio_mels_gt_dir):
            print(f"ERRORE: Directory mel GT audio custom non trovata: {opt.custom_audio_mels_gt_dir}")
            sys.exit(1)

        with open(opt.custom_data_list_txt, "r") as f:
            data_list1 = f.readlines()
            data_list1 = [x.strip() for x in data_list1 if x.strip()]

        spec_list1 = [os.path.join(opt.custom_audio_mels_gt_dir, base_name + "_mel.npy") for base_name in data_list1]
        # Assumiamo che le feature visive siano .npz con chiave 'feat', come per vggsound nello script originale
        video_list1 = [os.path.join(opt.custom_visual_features_dir, base_name + ".npz") for base_name in data_list1]
        # Se le tue feature visive sono .npy, commenta la riga sopra e decommenta questa:
        # video_list1 = [os.path.join(opt.custom_visual_features_dir, base_name + ".npy") for base_name in data_list1]

        print(f"Trovati {len(data_list1)} campioni custom da processare.")
        # Verifica esistenza file per il primo campione (debug)
        if data_list1:
            print(f"  Verifica per il primo campione '{data_list1[0]}':")
            print(f"    Path Mel GT: {spec_list1[0]} (Esiste? {os.path.exists(spec_list1[0])})")
            print(f"    Path Visual Feat: {video_list1[0]} (Esiste? {os.path.exists(video_list1[0])})")

    # Rimuoviamo la logica 'root' e i dataset hardcoded per VGGsound ecc.
    # Se vuoi testare con quelli, dovrai scaricare i loro dati e adattare i path.
    # Per ora, ci concentriamo sull'opzione 'custom'.
    elif opt.test_dataset in ['vggsound', 'landscape', 'Aist', 'yt4m']:
        print(
            f"ATTENZIONE: La logica per il dataset '{opt.test_dataset}' usa path hardcoded che probabilmente non funzioneranno.")
        print("  Questo script è stato modificato per favorire l'opzione '--test_dataset custom'.")
        print("  Se vuoi usare un dataset standard, dovrai scaricarlo e modificare i path qui.")
        # Qui potresti mettere la logica originale per quei dataset, ma con path modificabili
        # o dare direttamente errore. Per ora, diamo errore.
        sys.exit(f"Dataset '{opt.test_dataset}' non supportato con path locali in questa versione modificata.")
    else:
        print(f"ERRORE: --test_dataset '{opt.test_dataset}' non riconosciuto. Usa 'custom' o implementa altri dataset.")
        sys.exit(1)

    if not video_list1:  # Dovrebbe essere data_list1 qui
        print(
            f"ERRORE: Nessun campione da processare per test_dataset='{opt.test_dataset}'. Controlla i path e il file lista.")
        sys.exit(1)

    # Lettura parametri dal config YAML
    sr = opt.sample_rate

    cfg_data_train_dataset = config['data']['params']['train']['params']['dataset_cfg']
    duration = float(cfg_data_train_dataset['duration'])  # Assicura sia float
    fps = int(cfg_data_train_dataset['fps'])
    hop_len = int(cfg_data_train_dataset['hop_len'])

    # Parametri globali audio per il fallback di spec_raw (se il caricamento del mel GT fallisce)
    # Questi dovrebbero essere già stati usati per creare i tuoi mel GT.
    # Sarebbe meglio leggerli direttamente dal config come fatto nello script di estrazione.
    # Per ora, assumiamo che siano consistenti.
    # AUDIO_N_MELS = config['model']['params']['first_stage_config']['params']['ddconfig']['in_channels']
    # AUDIO_TARGET_SR_FOR_MEL_FALLBACK = config['data']['params']['train']['params']['dataset_cfg']['sr']
    # VIDEO_DURATION_SECONDS_FOR_MEL_FALLBACK = config['data']['params']['train']['params']['dataset_cfg']['duration']
    # AUDIO_HOP_LENGTH_FOR_MEL_FALLBACK = config['data']['params']['train']['params']['dataset_cfg']['hop_len']
    # Se AUDIO_N_MELS etc. non sono accessibili qui, dovrai passarli o definirli.
    # Per ora, usiamo valori fissi come nello script originale per il fallback.
    N_MELS_FALLBACK = 80  # Dovrebbe corrispondere a config['model']['params']['first_stage_config']['params']['ddconfig']['in_channels']
    MEL_LEN_FALLBACK_FRAMES = 625  # Corrisponde a 10s a 16kHz con hop 256 (16000 * 10 / 256)
    # Calcoliamolo dinamicamente se possibile:
    try:
        n_mels_from_cfg = int(config['model']['params']['first_stage_config']['params']['ddconfig']['in_channels'])
        sr_data_from_cfg = int(cfg_data_train_dataset['sr'])
        duration_data_from_cfg = float(cfg_data_train_dataset['duration'])
        hop_len_data_from_cfg = int(cfg_data_train_dataset['hop_len'])
        N_MELS_FALLBACK = n_mels_from_cfg
        MEL_LEN_FALLBACK_FRAMES = int(sr_data_from_cfg * duration_data_from_cfg / hop_len_data_from_cfg)
        print(f"Fallback Mel GT: N_MELS={N_MELS_FALLBACK}, LEN_FRAMES={MEL_LEN_FALLBACK_FRAMES}")
    except Exception as e_fallback_cfg:
        print(
            f"WARN: Impossibile leggere parametri per fallback Mel GT dal config: {e_fallback_cfg}. Uso default 80x625.")
        N_MELS_FALLBACK = 80
        MEL_LEN_FALLBACK_FRAMES = 625

    # Lunghezza della finestra per le feature visive
    visual_feat_seq_len = int(
        config['model']['params']['cond_stage_config']['params'].get('seq_len', int(fps * duration)))
    truncate_frame_visual = visual_feat_seq_len
    print(f"Lunghezza finestra per feature visive (truncate_frame_visual): {truncate_frame_visual}")

    uc = None  # Inizializza uc
    if opt.scale != 1.0:
        empty_vid_actual_path = opt.custom_empty_vid_path
        if not os.path.exists(empty_vid_actual_path):
            print(f"ERRORE: File empty_vid.npz non trovato in '{empty_vid_actual_path}'. Necessario per scale != 1.0.")
            sys.exit(1)
        print(f"Caricamento unconditional features da: {empty_vid_actual_path}")
        unconditional_np = np.load(empty_vid_actual_path)['feat'].astype(np.float32)

        expected_uncond_len = truncate_frame_visual
        if unconditional_np.shape[0] < expected_uncond_len:
            print(f"  Tiling unconditional features da {unconditional_np.shape[0]} a {expected_uncond_len} frames.")
            unconditional_np = np.tile(unconditional_np,
                                       (math.ceil(expected_uncond_len / unconditional_np.shape[0]), 1))
        unconditional_np = unconditional_np[:expected_uncond_len]

        unconditional = torch.from_numpy(unconditional_np).unsqueeze(0).to(device)
        uc = model.get_learned_conditioning(unconditional)
        print(f"Unconditional conditioning (uc) preparato. Input shape: {unconditional.shape}")

    output_mel_shape = None
    if opt.length is not None:
        output_mel_shape = (1, config['model']['params']['mel_dim'], opt.length)
        print(f"Override forma output mel a: {output_mel_shape}")
        # Logica NTK RoPE (come prima)
        from ldm.modules.diffusionmodules.flag_large_dit_moe import VideoFlagLargeDiT
        training_mel_length = config['model']['params']['mel_length']
        ntk_factor = opt.length // training_mel_length
        if ntk_factor > 1 and hasattr(model.model.diffusion_model, 'freqs_cis'):
            print(f"Adattamento RoPE per lunghezza output mel {opt.length} (ntk_factor={ntk_factor})...")
            dit_hidden_size = config['model']['params']['unet_config']['params']['hidden_size']
            dit_num_heads = config['model']['params']['unet_config']['params']['num_heads']
            dit_max_len_rope = config['model']['params']['unet_config']['params']['max_len']
            model.model.diffusion_model.freqs_cis = VideoFlagLargeDiT.precompute_freqs_cis(
                dit_hidden_size // dit_num_heads, dit_max_len_rope, ntk_factor=ntk_factor)
            print(f"RoPE (freqs_cis) aggiornate con ntk_factor={ntk_factor} per max_len_rope={dit_max_len_rope}.")
        # ... (altri print per ntk_factor)

    total_samples_to_process = len(data_list1)  # Usa data_list1 che è stata popolata
    print(f"\nInizio processamento di {total_samples_to_process} campioni...")

    for i_sample, sample_base_name in enumerate(data_list1):  # Itera sui nomi base
        current_spec_path = os.path.join(opt.custom_audio_mels_gt_dir, sample_base_name + "_mel.npy")
        current_video_feat_path = os.path.join(opt.custom_visual_features_dir, sample_base_name + ".npz")  # o .npy
        # Se usi .npy per le feature visive:
        # current_video_feat_path = os.path.join(opt.custom_visual_features_dir, sample_base_name + ".npy")

        name_stem = sample_base_name  # Usiamo il nome base per i file di output
        print(f"\nProcessando campione {i_sample + 1}/{total_samples_to_process}: {name_stem}")
        print(f"  Mel GT da: {current_spec_path}")
        print(f"  Visual Feat da: {current_video_feat_path}")

        if os.path.exists(os.path.join(opt.outdir, name_stem + f'_0.wav')):
            print(f"  Output già esistente, skipping: {name_stem}")
            continue

        try:
            spec_raw_np = np.load(current_spec_path).astype(np.float32)
        except Exception as e_load_spec:
            print(f"  ERRORE: Impossibile caricare mel GT da '{current_spec_path}': {e_load_spec}. Uso zeri.")
            spec_raw_np = np.zeros((N_MELS_FALLBACK, MEL_LEN_FALLBACK_FRAMES), dtype=np.float32)
        print(f"  Mel GT caricato/fallback. Shape: {spec_raw_np.shape}")

        try:
            if current_video_feat_path.endswith(".npz"):
                video_feat_np = np.load(current_video_feat_path)['feat'].astype(np.float32)
            elif current_video_feat_path.endswith(".npy"):
                video_feat_np = np.load(current_video_feat_path).astype(np.float32)
            else:
                raise ValueError(f"Formato file non riconosciuto: {current_video_feat_path}")
        except Exception as e_load_vf:
            print(f"  ERRORE CRITICO caricando feature visive da '{current_video_feat_path}': {e_load_vf}. Skipping.")
            continue
        print(f"  Feature visive caricate. Shape: {video_feat_np.shape}")

        spec_len_frames_expected = int(sr * duration / hop_len)
        if spec_raw_np.shape[1] < spec_len_frames_expected:
            spec_raw_np = np.tile(spec_raw_np, (1, math.ceil(spec_len_frames_expected / spec_raw_np.shape[1])))
        spec_raw_np = spec_raw_np[:, :spec_len_frames_expected]

        feat_len_frames_expected = int(fps * duration)
        if video_feat_np.shape[0] < feat_len_frames_expected:
            video_feat_np = np.tile(video_feat_np, (math.ceil(feat_len_frames_expected / video_feat_np.shape[0]), 1))
        video_feat_np = video_feat_np[:feat_len_frames_expected]

        spec_raw_torch = torch.from_numpy(spec_raw_np).unsqueeze(0).to(device)
        video_feat_torch = torch.from_numpy(video_feat_np).unsqueeze(0).to(device)
        print(f"  Mel GT (tensor) shape: {spec_raw_torch.shape}, Visual Feat (tensor) shape: {video_feat_torch.shape}")

        current_video_feat_len_frames = video_feat_torch.shape[1]
        window_num = math.ceil(current_video_feat_len_frames / truncate_frame_visual)
        print(f"  Numero di finestre da processare (truncate_frame_visual={truncate_frame_visual}): {window_num}")

        gt_mel_chunks_list, generated_mel_chunks_list = [], []
        for i_window in tqdm(range(window_num), desc=f"    Finestre per {name_stem}"):
            vf_start = i_window * truncate_frame_visual
            vf_end = min((i_window + 1) * truncate_frame_visual, current_video_feat_len_frames)
            current_video_feat_chunk = video_feat_torch[:, vf_start:vf_end]

            if current_video_feat_chunk.shape[
                1] < truncate_frame_visual and opt.scale != 1.0:  # Padding solo se si usa CFG e il chunk è corto
                padding_size = truncate_frame_visual - current_video_feat_chunk.shape[1]
                padding_vf = torch.zeros(current_video_feat_chunk.shape[0], padding_size,
                                         current_video_feat_chunk.shape[2]).to(device)
                current_video_feat_chunk = torch.cat([current_video_feat_chunk, padding_vf], dim=1)

            audio_frames_per_visual_frame = (sr / hop_len) / fps
            mel_gt_start = int(vf_start * audio_frames_per_visual_frame)
            mel_gt_end = int(vf_end * audio_frames_per_visual_frame)
            mel_gt_end = min(mel_gt_end, spec_raw_torch.shape[2])
            current_gt_mel_chunk = spec_raw_torch[:, :, mel_gt_start:mel_gt_end]

            c = model.get_learned_conditioning(current_video_feat_chunk)

            current_output_mel_shape_window = None  # Inizializza per questa finestra
            if output_mel_shape is not None:  # Se opt.length è dato, usiamo una frazione di esso per la finestra
                # La lunghezza mel per questa finestra dovrebbe corrispondere a current_gt_mel_chunk
                # se opt.length non è specificato. Se è specificato, dobbiamo calcolare la
                # lunghezza corrispondente per questo chunk.
                # Questo è complesso se opt.length non è un multiplo di window_num.
                # Per CFM, la lunghezza è spesso determinata dall'input condizionante.
                # Se usiamo opt.length, dobbiamo capire come il modello lo gestisce per i chunk.
                # Per ora, se opt.length è dato, passiamolo direttamente (il modello potrebbe troncare/paddare).
                # Questa parte potrebbe necessitare di un'ulteriore revisione se l'output non ha la lunghezza attesa.
                # Lo script originale sembrava usare 'shape' (output_mel_shape) per l'intera sequenza.
                # Ma qui processiamo a finestre.
                # Se opt.length è dato, è per l'INTERA sequenza.
                # Quindi, per il chunk, la lunghezza dovrebbe essere la lunghezza del GT di questo chunk.
                if current_gt_mel_chunk.shape[2] > 0:
                    current_output_mel_shape_window = (
                    1, config['model']['params']['mel_dim'], current_gt_mel_chunk.shape[2])
            # Se current_output_mel_shape_window è ancora None, significa che il modello deciderà la lunghezza
            # basandosi su 'c' o la sua lunghezza di addestramento se 'shape' è None in model.sample/sample_cfg.

            if opt.scale == 1.0:
                sample, _ = model.sample(c, 1, timesteps=opt.ddim_steps, shape=current_output_mel_shape_window)
            else:
                sample, _ = model.sample_cfg(c, opt.scale, uc, 1, timesteps=opt.ddim_steps,
                                             shape=current_output_mel_shape_window)

            x_samples_ddim = model.decode_first_stage(sample)

            # Tronca/padda l'output per farlo corrispondere al GT del chunk se necessario
            if current_gt_mel_chunk.shape[2] > 0:
                if x_samples_ddim.shape[2] > current_gt_mel_chunk.shape[2]:
                    x_samples_ddim = x_samples_ddim[:, :, :current_gt_mel_chunk.shape[2]]
                elif x_samples_ddim.shape[2] < current_gt_mel_chunk.shape[2]:
                    padding_mel = torch.zeros(x_samples_ddim.shape[0], x_samples_ddim.shape[1],
                                              current_gt_mel_chunk.shape[2] - x_samples_ddim.shape[2]).to(device)
                    x_samples_ddim = torch.cat([x_samples_ddim, padding_mel], dim=2)

                generated_mel_chunks_list.append(x_samples_ddim)
                gt_mel_chunks_list.append(current_gt_mel_chunk)
            elif x_samples_ddim.shape[2] > 0:  # Se non c'è GT ma abbiamo generato qualcosa
                generated_mel_chunks_list.append(x_samples_ddim)
                # Non possiamo aggiungere a gt_mel_chunks_list, ma potremmo voler salvare l'audio generato

        syn_mel_full, gt_mel_full = None, None
        if generated_mel_chunks_list:
            syn_mel_full = torch.cat(generated_mel_chunks_list, dim=2)
            print(f"  Melodia sintetizzata completa. Shape: {syn_mel_full.shape}")
        if gt_mel_chunks_list:  # Solo se abbiamo aggiunto chunk GT
            gt_mel_full = torch.cat(gt_mel_chunks_list, dim=2)
            print(f"  Melodia GT completa. Shape: {gt_mel_full.shape}")

        if gt_mel_full is not None and gt_mel_full.shape[2] > 0:
            spec_gt_for_vocoder = gt_mel_full.squeeze(0).cpu().numpy()
            wav_gt = vocoder.vocode(spec_gt_for_vocoder)
            wav_path_gt = os.path.join(opt.outdir, name_stem + f'_0_gt.wav')
            soundfile.write(wav_path_gt, wav_gt, opt.sample_rate)
            print(f"  Audio GT salvato: {wav_path_gt}")

        if syn_mel_full is not None and syn_mel_full.shape[2] > 0:
            spec_syn_for_vocoder = syn_mel_full.squeeze(0).cpu().numpy()
            ddim_wav = vocoder.vocode(spec_syn_for_vocoder)
            wav_path_syn = os.path.join(opt.outdir, name_stem + f'_0.wav')
            soundfile.write(wav_path_syn, ddim_wav, opt.sample_rate)
            print(f"  Audio sintetizzato salvato: {wav_path_syn}")
        elif syn_mel_full is None:
            print(f"  ATTENZIONE: Nessun mel sintetizzato generato per {name_stem}.")

    print(f"\nI tuoi campioni sono pronti e ti aspettano qui: \n{opt.outdir} \nEnjoy.")


if __name__ == "__main__":
    main()