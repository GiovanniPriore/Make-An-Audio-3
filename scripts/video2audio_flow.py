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
    opt = parse_args()

    # Stampa argomenti per debug
    print("--- ARGOMENTI PARSATI ---")
    for arg, value in vars(opt).items():
        print(f"{arg}: {value}")
    print("-------------------------")

    config = OmegaConf.load(opt.base)

    # Verifica se il checkpoint del modello principale è specificato e valido
    if not opt.resume or not os.path.exists(opt.resume):
        print(f"ERRORE: Checkpoint del modello V2A non specificato o non trovato: {opt.resume}")
        sys.exit(1)

    model = load_model_from_config(config, opt.resume)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    os.makedirs(opt.outdir, exist_ok=True)

    # Gestione path vocoder
    # L'argomento da riga di comando --vocoder-ckpt (se lo aggiungi) dovrebbe avere la precedenza.
    # Altrimenti, usa quello dal config.
    # Per ora, assumiamo che il config YAML punti a un path valido o che tu copi i file.
    # Path originale dal config: config['lightning']['callbacks']['image_logger']['params']['vocoder_cfg']['params']['ckpt_vocoder']
    # Questo path nel config YAML è: ldm_src/ckpt/bigvnat
    # Se vuoi usare un path diverso passato da riga di comando, dovresti aggiungere un argomento --vocoder_ckpt
    # e usarlo qui. Per ora, usiamo il path dal config, assumendo che tu abbia copiato i file vocoder
    # in /kaggle/working/Make-An-Audio-3/ldm_src/ckpt/bigvnat/ (es. g_xxxxxxxx e config.json)
    # OPPURE, se hai caricato i file del vocoder nel tuo dataset Kaggle, modifica questo path.
    vocoder_ckpt_path_from_config = config['lightning']['callbacks']['image_logger']['params']['vocoder_cfg']['params'][
        'ckpt_vocoder']
    # Esempio: se i file del vocoder (generatore .pth e config.json) sono in /kaggle/input/dataset-maa/vocoder_files/
    # e il generatore si chiama g_model.pth, allora:
    # vocoder_ckpt_path_to_use = "/kaggle/input/dataset-maa/vocoder_files/g_model.pth"
    # Per ora, assumiamo che l'utente prepari la directory ldm_src/ckpt/bigvnat/

    # Prepara il path completo al file del generatore del vocoder.
    # VocoderBigVGAN si aspetta il path al file del generatore .pth,
    # e il config.json deve essere nella stessa directory.
    # Il tuo YAML per V2A ha 'ldm_src/ckpt/bigvnat'. Assumiamo che questo sia il NOME BASE.
    # Se il file è g_02500000.pth, allora il path base è 'ldm_src/ckpt/bigvnat/g_02500000'
    # E il config.json è in 'ldm_src/ckpt/bigvnat/config.json'
    # Per ora, usiamo un path placeholder che dovrai sistemare:

    # Assumiamo che la directory 'ldm_src/ckpt/bigvnat/' contenga 'g_model.pth' e 'config.json'
    # (dovrai scaricarli e metterli lì, o nel tuo dataset Kaggle e cambiare il path)
    # Esempio: path_to_vocoder_generator_file = "/kaggle/working/Make-An-Audio-3/ldm_src/ckpt/bigvnat/g_02500000.pth"
    # Se hai caricato g_02500000.pth e config.json in /kaggle/input/dataset-maa/vocoder_bigvnat/
    path_to_vocoder_generator_file = "/kaggle/input/dataset-maa/Dataset_MAA/Dataset_MAA/CLAP_weights_2022.pth"  # MODIFICA QUESTO SE NECESSARIO

    if not os.path.exists(path_to_vocoder_generator_file):
        print(f"ERRORE: File generatore Vocoder '{path_to_vocoder_generator_file}' non trovato!")
        print("Assicurati di aver scaricato g_xxxx.pth e config.json e che il path sia corretto.")
        sys.exit(1)
    if not os.path.exists(os.path.join(os.path.dirname(path_to_vocoder_generator_file), "config.json")):
        print(
            f"ERRORE: File config.json del Vocoder non trovato in '{os.path.dirname(path_to_vocoder_generator_file)}'!")
        sys.exit(1)

    vocoder = VocoderBigVGAN(path_to_vocoder_generator_file, device)
    print(f"Vocoder caricato da: {path_to_vocoder_generator_file}")

    # Rimuovi la logica di 'root' hardcoded
    # if os.path.exists('/apdcephfs/share_1316500/nlphuang/data/video_to_audio/vggsound/split_txt'):
    #     root = '/apdcephfs'
    # else:
    #     root = '/apdcephfs_intern'

    spec_list1 = []
    video_list1 = []

    ### MODIFICA START: Logica per caricare il dataset "custom" ###
    if opt.test_dataset == 'custom':
        print(f"Caricamento dataset custom dai path forniti:")
        print(f"  Lista file: {opt.custom_data_list_txt}")
        print(f"  Dir feature visive: {opt.custom_visual_features_dir}")
        print(f"  Dir mel GT audio: {opt.custom_audio_mels_gt_dir}")

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
            data_list1 = [x.strip() for x in data_list1 if x.strip()]  # Rimuovi whitespace e righe vuote

        # Costruisci i path completi ai file di feature
        # Assumiamo che data_list1 contenga i nomi base dei file (es. "video1", "video2")
        spec_list1 = [os.path.join(opt.custom_audio_mels_gt_dir, base_name + "_mel.npy") for base_name in data_list1]
        # Adatta l'estensione per le visual features se usi .npy invece di .npz
        video_list1 = [os.path.join(opt.custom_visual_features_dir, base_name + ".npz") for base_name in data_list1]
        # Se salvi le visual features come .npy:
        # video_list1 = [os.path.join(opt.custom_visual_features_dir, base_name + ".npy") for base_name in data_list1]

        # Verifica che i file esistano (opzionale, ma buon debug)
        for p in spec_list1 + video_list1:
            if not os.path.exists(p):
                print(f"ATTENZIONE: File di feature atteso non trovato: {p}")
                # Potresti voler rimuovere il campione dalla lista o dare errore
    ### MODIFICA END ###

    elif opt.test_dataset == 'vggsound':
        # Manteniamo la logica originale ma con un warning sui path hardcoded
        print(
            "ATTENZIONE: Stai usando il dataset 'vggsound' con path hardcoded. Questo fallirà se non sei nell'ambiente originale.")
        root = '/apdcephfs_intern'  # Scegli uno dei root originali, probabilmente fallirà comunque
        split, data = f'{root}/share_1316500/nlphuang/data/video_to_audio/vggsound/split_txt', f'{root}/share_1316500/nlphuang/data/video_to_audio/vggsound/'
        dataset1_spec_dir = os.path.join(data, "mel_maa2", "npy")
        dataset1_feat_dir = os.path.join(data, "cavp")

        with open(os.path.join(split, 'vggsound_test.txt'), "r") as f:
            data_list1 = f.readlines()
            data_list1 = list(map(lambda x: x.strip(), data_list1))
            spec_list1 = list(map(lambda x: os.path.join(dataset1_spec_dir, x) + "_mel.npy", data_list1))
            video_list1 = list(map(lambda x: os.path.join(dataset1_feat_dir, x) + ".npz", data_list1))

    # Aggiungi qui gli altri blocchi elif per 'landscape', 'Aist', 'yt4m' se vuoi mantenerli,
    # ma probabilmente daranno errore a causa dei path /apdcephfs.
    # Per ora, li omettiamo per brevità, concentrandoci su 'custom'.

    else:
        # Modificato per dare un errore più specifico se non è 'custom' o un altro dataset noto.
        if opt.test_dataset not in ['vggsound', 'landscape', 'Aist', 'yt4m']:  # Aggiungi altri se li implementi
            print(
                f"ERRORE: --test_dataset '{opt.test_dataset}' non supportato o non implementato correttamente in questo script modificato.")
        else:
            print(
                f"ERRORE: Logica per --test_dataset '{opt.test_dataset}' non completamente implementata o usa path hardcoded.")
        sys.exit(1)  # Esce se il dataset non è gestito

    if not video_list1:
        print(
            f"ERRORE: Nessun file video/feature da processare per test_dataset='{opt.test_dataset}'. Controlla i path e il file lista.")
        sys.exit(1)

    # Lettura parametri dal config (già fatta in parte nello script di estrazione, ma utile averli qui)
    sr = opt.sample_rate  # Usa quello da riga di comando per l'output wav

    # Questi vengono dal config YAML del modello V2A
    cfg_data_train_dataset = config['data']['params']['train']['params']['dataset_cfg']
    duration = cfg_data_train_dataset['duration']
    # truncate = cfg_data_train_dataset['truncate'] # 'truncate' qui è in campioni audio, non frame visivi
    fps = cfg_data_train_dataset['fps']
    hop_len = cfg_data_train_dataset['hop_len']

    # 'truncate' nel config YAML si riferisce a campioni audio (es. 131072).
    # Lo script originale calcola truncate_frame per le feature visive in un modo un po' strano:
    # truncate_frame = int(fps * truncate / sr)
    # Questo sembra voler allineare un troncamento audio a frame visivi.
    # Per CFM, il modello è un ODE solver, potrebbe non aver bisogno di questo troncamento rigido
    # se gestisce sequenze di lunghezza variabile o se la lunghezza è fissata da 'opt.length'.
    # Per il momento, manteniamo una logica di troncamento simile se opt.length non è specificato,
    # usando una lunghezza di finestra basata su seq_len del cond_stage_config.

    # Lunghezza della sequenza di feature visive attesa dall'encoder del modello
    visual_feat_seq_len = config['model']['params']['cond_stage_config']['params'].get('seq_len', int(fps * duration))
    # Se opt.length è specificato (lunghezza mel output), dobbiamo adattare la finestra
    # Se non specificato, processiamo l'intera durata (o la seq_len dell'encoder visivo)

    # truncate_frame è la dimensione della finestra per processare le feature visive.
    # Se opt.length (per l'output mel) non è specificato, usiamo visual_feat_seq_len come dimensione della finestra.
    # Se opt.length è specificato, il modello UNet/DiT potrebbe essere configurato per quella lunghezza.
    # La logica originale per 'truncate_frame' era confusa. Semplifichiamo:
    # Se il modello ha un max_len per le feature visive (cond_stage_config.params.seq_len), usiamo quello.
    truncate_frame_visual = visual_feat_seq_len  # Numero di frame visivi da processare alla volta

    if opt.scale != 1:
        ### MODIFICA START: Usa il path custom per empty_vid.npz ###
        empty_vid_actual_path = opt.custom_empty_vid_path
        if not os.path.exists(empty_vid_actual_path):
            print(f"ERRORE: File empty_vid.npz non trovato in '{empty_vid_actual_path}'. Necessario per scale != 1.0.")
            sys.exit(1)
        print(f"Caricamento unconditional features da: {empty_vid_actual_path}")
        unconditional_np = np.load(empty_vid_actual_path)['feat'].astype(np.float32)
        ### MODIFICA END ###

        # La lunghezza attesa per le feature incondizionate dovrebbe corrispondere a truncate_frame_visual
        expected_uncond_len = truncate_frame_visual
        if unconditional_np.shape[0] < expected_uncond_len:
            print(f"Tiling unconditional features da {unconditional_np.shape[0]} a {expected_uncond_len} frames.")
            unconditional_np = np.tile(unconditional_np,
                                       (math.ceil(expected_uncond_len / unconditional_np.shape[0]), 1))
        unconditional_np = unconditional_np[:expected_uncond_len]

        unconditional = torch.from_numpy(unconditional_np).unsqueeze(0).to(device)
        # Non c'è bisogno di un ulteriore troncamento qui se unconditional_np ha già la lunghezza truncate_frame_visual
        # unconditional = unconditional[:, :truncate_frame_visual] # Originale era :truncate_frame
        uc = model.get_learned_conditioning(unconditional)
        print(f"Unconditional conditioning (uc) preparato. Shape: {unconditional.shape}")
    else:
        uc = None  # Assicurati che uc sia definito

    # Gestione della lunghezza della sequenza di output mel (se specificata)
    output_mel_shape = None
    if opt.length is not None:  # opt.length è in numero di frame mel
        output_mel_shape = (1, config['model']['params']['mel_dim'], opt.length)
        print(f"Override forma output mel a: {output_mel_shape}")

        # La parte di ntk_factor per DiT a lunghezza variabile
        # Questa parte adatta le embedding posizionali del DiT se opt.length è diverso da mel_length di addestramento
        from ldm.modules.diffusionmodules.flag_large_dit_moe import VideoFlagLargeDiT  # Assicurati sia il DiT corretto

        # mel_length di addestramento (dal config del modello, non dei dati)
        training_mel_length = config['model']['params']['mel_length']
        ntk_factor = opt.length // training_mel_length

        if ntk_factor > 1 and hasattr(model.model.diffusion_model, 'freqs_cis'):  # Controlla se il DiT ha freqs_cis
            print(f"Adattamento RoPE per lunghezza output mel {opt.length} (ntk_factor={ntk_factor})...")
            dit_hidden_size = config['model']['params']['unet_config']['params']['hidden_size']
            dit_num_heads = config['model']['params']['unet_config']['params']['num_heads']
            # max_len qui è la max_len del DiT, non necessariamente opt.length.
            # Dovrebbe essere la lunghezza massima per cui le RoPE sono state precalcolate o possono essere estese.
            # Usiamo la max_len definita nel config dell'UNet.
            dit_max_len_rope = config['model']['params']['unet_config']['params']['max_len']

            # Se opt.length > dit_max_len_rope, il ntk scaling potrebbe essere applicato a una base
            # di lunghezza dit_max_len_rope, e poi si spera che il modello generalizzi.
            # O, idealmente, max_len per precompute_freqs_cis dovrebbe essere la nuova opt.length.
            # Per ora, usiamo dit_max_len_rope come base per lo scaling ntk.

            model.model.diffusion_model.freqs_cis = VideoFlagLargeDiT.precompute_freqs_cis(
                dit_hidden_size // dit_num_heads,
                dit_max_len_rope,
                # Max len per cui calcolare le RoPE, potrebbe essere opt.length se si vuole precisione
                ntk_factor=ntk_factor
            )
            print(
                f"RoPE (freqs_cis) nel DiT aggiornate con ntk_factor={ntk_factor} per max_len_rope={dit_max_len_rope}.")
        elif ntk_factor <= 1:
            print(
                f"Lunghezza output mel ({opt.length}) <= lunghezza di addestramento ({training_mel_length}). Nessun scaling NTK RoPE necessario.")
        else:
            print(
                f"WARN: Impossibile applicare scaling NTK RoPE (DiT potrebbe non averle o opt.length non è significativamente maggiore).")

    total_samples_to_process = len(spec_list1)
    print(f"Inizio processamento di {total_samples_to_process} campioni...")

    for i_sample, (spec_path, video_feat_path) in enumerate(zip(spec_list1, video_list1)):
        name = Path(video_feat_path).stem
        print(f"\nProcessando campione {i_sample + 1}/{total_samples_to_process}: {name}")

        # Skip se l'output esiste già (utile per riprendere run interrotte)
        # Controlla solo il file .wav generato, non il _gt.wav
        if os.path.exists(os.path.join(opt.outdir, name + f'_0.wav')):  # Assumendo idx=0 se non c'è loop n_samples
            print(f"  Output già esistente, skipping: {name}")
            continue

        # Caricamento mel-spettrogramma ground truth (spec_raw)
        try:
            spec_raw_np = np.load(spec_path).astype(np.float32)
            print(f"  Mel GT caricato da '{spec_path}'. Shape: {spec_raw_np.shape}")
        except Exception as e_load_spec:
            print(f"  ERRORE: Impossibile caricare mel GT da '{spec_path}': {e_load_spec}. Uso zeri.")
            # Calcola la lunghezza attesa dei frame mel per il ground truth
            # expected_mel_gt_len_frames = int(sr * duration / hop_len) # sr, duration, hop_len dal config dati
            # Per ora, usiamo una shape di fallback fissa se il caricamento fallisce, come l'originale.
            # Questo dovrebbe essere raro se lo script di estrazione mel funziona.
            # La shape originale era (80, 625). 80 è N_MELS. 625 frames * 256 hop / 16000 sr = 10 secondi.
            # Quindi 625 è corretto per 10s, 16kHz, hop 256.
            expected_mel_gt_len_frames = int(
                AUDIO_TARGET_SR * VIDEO_DURATION_SECONDS / AUDIO_HOP_LENGTH)  # Usa le var globali definite per l'estrazione audio
            spec_raw_np = np.zeros((AUDIO_N_MELS, expected_mel_gt_len_frames), dtype=np.float32)
            print(f"  Usando mel GT di zeri. Shape: {spec_raw_np.shape}")

        # Caricamento feature visive (video_feat)
        try:
            # Adatta questo se hai salvato come .npy invece di .npz
            if video_feat_path.endswith(".npz"):
                video_feat_np = np.load(video_feat_path)['feat'].astype(np.float32)
            elif video_feat_path.endswith(".npy"):
                video_feat_np = np.load(video_feat_path).astype(np.float32)
            else:
                raise ValueError(f"Formato file feature visive non riconosciuto: {video_feat_path}")
            print(f"  Feature visive caricate da '{video_feat_path}'. Shape: {video_feat_np.shape}")
        except Exception as e_load_vf:
            print(f"  ERRORE CRITICO: Impossibile caricare feature visive da '{video_feat_path}': {e_load_vf}.")
            print(f"  Skipping campione {name}.")
            continue  # Salta al prossimo campione

        # Tiling/Padding per spec_raw (GT mel)
        # spec_len è il numero di frame mel attesi per la 'duration' completa
        spec_len_frames_expected = int(
            sr * duration / hop_len)  # sr è opt.sample_rate, duration e hop_len dal config dati
        if spec_raw_np.shape[1] < spec_len_frames_expected:
            print(f"  Tiling mel GT da {spec_raw_np.shape[1]} a {spec_len_frames_expected} frames.")
            spec_raw_np = np.tile(spec_raw_np, (1, math.ceil(spec_len_frames_expected / spec_raw_np.shape[1])))
        spec_raw_np = spec_raw_np[:, :spec_len_frames_expected]
        print(f"  Mel GT (spec_raw) final shape: {spec_raw_np.shape}")

        # Tiling/Padding per video_feat
        # feat_len_frames_expected è il numero di frame visivi attesi (es. 40 per 10s a 4fps)
        feat_len_frames_expected = int(fps * duration)  # fps e duration dal config dati
        if video_feat_np.shape[0] < feat_len_frames_expected:
            print(f"  Tiling visual features da {video_feat_np.shape[0]} a {feat_len_frames_expected} frames.")
            video_feat_np = np.tile(video_feat_np, (math.ceil(feat_len_frames_expected / video_feat_np.shape[0]), 1))
        video_feat_np = video_feat_np[:feat_len_frames_expected]
        print(f"  Visual features (video_feat) final shape: {video_feat_np.shape}")

        # Conversione a tensori PyTorch
        spec_raw_torch = torch.from_numpy(spec_raw_np).unsqueeze(0).to(device)
        video_feat_torch = torch.from_numpy(video_feat_np).unsqueeze(0).to(device)

        # Logica di windowing per processare video lunghi a pezzi (usando truncate_frame_visual)
        # Se il video è già della lunghezza giusta (truncate_frame_visual), window_num sarà 1.
        current_video_feat_len_frames = video_feat_torch.shape[1]
        window_num = math.ceil(
            current_video_feat_len_frames / truncate_frame_visual)  # Usa ceil per coprire l'ultimo pezzo
        print(
            f"  Numero di finestre da processare (basato su truncate_frame_visual={truncate_frame_visual}): {window_num}")

        gt_mel_chunks_list, generated_mel_chunks_list = [], []

        for i_window in tqdm(range(window_num), desc=f"  Finestre per {name}"):
            # Calcola start/end per le feature visive per questa finestra
            vf_start = i_window * truncate_frame_visual
            vf_end = min((i_window + 1) * truncate_frame_visual, current_video_feat_len_frames)
            current_video_feat_chunk = video_feat_torch[:, vf_start:vf_end]

            # Se l'ultimo chunk è più corto di truncate_frame_visual, fai padding (necessario per 'uc')
            if current_video_feat_chunk.shape[1] < truncate_frame_visual:
                padding_size = truncate_frame_visual - current_video_feat_chunk.shape[1]
                padding = torch.zeros(current_video_feat_chunk.shape[0], padding_size,
                                      current_video_feat_chunk.shape[2]).to(device)
                current_video_feat_chunk = torch.cat([current_video_feat_chunk, padding], dim=1)
                print(f"    Finestra {i_window}: Chunk visivo paddato a shape {current_video_feat_chunk.shape}")

            # Calcola start/end corrispondenti per lo spettrogramma audio GT
            # Questo allinea l'audio GT alla finestra delle feature visive
            # (sr e hop_len sono per l'audio, fps per il video)
            audio_frames_per_visual_frame = (sr / hop_len) / fps

            mel_gt_start = int(vf_start * audio_frames_per_visual_frame)
            mel_gt_end = int(vf_end * audio_frames_per_visual_frame)
            # Assicurati che mel_gt_end non superi la lunghezza di spec_raw_torch
            mel_gt_end = min(mel_gt_end, spec_raw_torch.shape[2])
            current_gt_mel_chunk = spec_raw_torch[:, :, mel_gt_start:mel_gt_end]

            print(
                f"    Finestra {i_window}: Chunk visivo shape {current_video_feat_chunk.shape}, Chunk Mel GT shape {current_gt_mel_chunk.shape}")

            # Ottieni condizionamento dalle feature visive del chunk corrente
            c = model.get_learned_conditioning(current_video_feat_chunk)

            # Determina la shape di output per questa finestra
            # Se output_mel_shape (basato su opt.length) è specificato, usalo.
            # Altrimenti, la shape di output dovrebbe corrispondere alla lunghezza del chunk GT mel.
            current_output_mel_shape = output_mel_shape
            if current_output_mel_shape is None:  # Se opt.length non è specificato
                # La shape dell'output mel generato dovrebbe corrispondere a quella del GT per questa finestra
                if current_gt_mel_chunk.shape[2] > 0:  # Solo se ci sono frame GT
                    current_output_mel_shape = (1, config['model']['params']['mel_dim'], current_gt_mel_chunk.shape[2])
                else:  # Se non ci sono frame GT (es. vf_end è troppo piccolo), salta o genera una lunghezza minima
                    print(
                        f"    WARN: Finestra {i_window} non ha frame Mel GT corrispondenti. Potrebbe generare audio vuoto o di lunghezza default.")
                    # Potresti decidere di saltare questa finestra se non ci sono frame audio GT
                    # o usare una lunghezza minima/default se il modello può gestirla.
                    # Per ora, se non c'è GT, non possiamo definire current_output_mel_shape basato su di esso.
                    # Il modello.sample potrebbe usare una sua lunghezza default se shape=None.
                    # Per CFM, la lunghezza è solitamente definita dalla ODE.
                    # Lo script originale non gestiva shape=None per CFM in modo dinamico basato sul chunk.
                    # Se opt.length non è dato, la lunghezza dell'output è determinata dalla lunghezza
                    # delle feature di condizionamento passate all'ODE (implicito in cfm1_audio.py).
                    # Il modello dovrebbe generare una lunghezza coerente con current_video_feat_chunk.
                    pass  # Lascia current_output_mel_shape a None, il modello deciderà.

            print(f"    Finestra {i_window}: Sampling con output_mel_shape: {current_output_mel_shape}")
            if opt.scale == 1:  # w/o cfg
                sample, _ = model.sample(c, 1, timesteps=opt.ddim_steps, shape=current_output_mel_shape)
            else:  # cfg
                if uc is None:
                    print(
                        "ERRORE: 'uc' (unconditional conditioning) è None ma scale != 1.0. Assicurati che empty_vid.npz sia processato.")
                    sys.exit(1)
                sample, _ = model.sample_cfg(c, opt.scale, uc, 1, timesteps=opt.ddim_steps,
                                             shape=current_output_mel_shape)

            x_samples_ddim = model.decode_first_stage(sample)

            # Se abbiamo paddato current_video_feat_chunk, l'output x_samples_ddim potrebbe essere più lungo
            # del current_gt_mel_chunk. Tronchiamo x_samples_ddim per farlo corrispondere a current_gt_mel_chunk
            # se current_gt_mel_chunk ha una lunghezza valida.
            if current_gt_mel_chunk.shape[2] > 0 and x_samples_ddim.shape[2] > current_gt_mel_chunk.shape[2]:
                print(
                    f"    Finestra {i_window}: Tronco mel generato da {x_samples_ddim.shape[2]} a {current_gt_mel_chunk.shape[2]} frames.")
                x_samples_ddim = x_samples_ddim[:, :, :current_gt_mel_chunk.shape[2]]

            if current_gt_mel_chunk.shape[2] > 0:  # Aggiungi solo se c'è GT
                generated_mel_chunks_list.append(x_samples_ddim)
                gt_mel_chunks_list.append(current_gt_mel_chunk)

        # Concatena i chunk per formare gli spettrogrammi completi
        syn_mel_full = None
        gt_mel_full = None

        if len(generated_mel_chunks_list) > 0:
            # Usa dim=2 per concatenare lungo l'asse temporale (0=batch, 1=mel_bins, 2=time_frames)
            syn_mel_full = torch.cat(generated_mel_chunks_list, dim=2)
            print(f"  Melodia sintetizzata completa concatenata. Shape: {syn_mel_full.shape}")
        if len(gt_mel_chunks_list) > 0:
            gt_mel_full = torch.cat(gt_mel_chunks_list, dim=2)
            print(f"  Melodia GT completa concatenata. Shape: {gt_mel_full.shape}")

        # Se non sono stati generati chunk (es. video troppo corto o problemi)
        if syn_mel_full is None or gt_mel_full is None:
            print(f"  ATTENZIONE: Non sono stati generati chunk validi per {name}. Salto la generazione audio.")
            continue

        # Il loop originale faceva enumerate(zip(gt_mel, syn_mel))
        # ma gt_mel e syn_mel erano già gli spettrogrammi completi (batch size 1).
        # Ora abbiamo syn_mel_full e gt_mel_full che sono [1, mel_bins, total_time_frames]

        # Processa l'intero spettrogramma GT concatenato
        # Rimuovi la dimensione batch prima di passare al vocoder se si aspetta [mel_bins, time_frames]
        spec_gt_for_vocoder = gt_mel_full.squeeze(0).cpu().numpy()
        wav_gt = vocoder.vocode(spec_gt_for_vocoder)
        wav_path_gt = os.path.join(opt.outdir, name + f'_0_gt.wav')  # idx è 0 perché processiamo l'intero campione
        soundfile.write(wav_path_gt, wav_gt, opt.sample_rate)
        print(f"  Audio GT salvato: {wav_path_gt}")

        # Processa l'intero spettrogramma sintetizzato concatenato
        spec_syn_for_vocoder = syn_mel_full.squeeze(0).cpu().numpy()
        ddim_wav = vocoder.vocode(spec_syn_for_vocoder)
        wav_path_syn = os.path.join(opt.outdir, name + f'_0.wav')  # idx è 0
        soundfile.write(wav_path_syn, ddim_wav, opt.sample_rate)
        print(f"  Audio sintetizzato salvato: {wav_path_syn}")

    print(f"\nI tuoi campioni sono pronti e ti aspettano qui: \n{opt.outdir} \nEnjoy.")


if __name__ == "__main__":
    # Definisci qui le variabili globali per i parametri audio se necessario
    # (se non sono già definite quando chiami main() da un altro script/notebook)
    # Esempio:
    # AUDIO_TARGET_SR = 16000
    # VIDEO_DURATION_SECONDS = 10
    # AUDIO_HOP_LENGTH = 256
    # AUDIO_N_MELS = 80
    # ... ecc.
    # Queste sarebbero idealmente lette dal config YAML all'inizio dello script main() o globalmente.
    # Per ora, assumiamo che la logica di caricamento config in main() le renda disponibili
    # o che lo script di estrazione mel le imposti globalmente se eseguito prima.

    # Per eseguire questo script da riga di comando, dovrai passare gli argomenti corretti, es:
    # python tuo_script_v2a_modificato.py \
    #   -b /path/al/tuo/video2audio-cfm-cfg-moe.yaml \
    #   -r /path/al/tuo/checkpoint_v2a.ckpt \
    #   --test_dataset custom \
    #   --custom_data_list_txt /kaggle/working/my_v2a_test_data/my_test_list.txt \
    #   --custom_visual_features_dir /kaggle/working/my_v2a_test_data/visual_features/ \
    #   --custom_audio_mels_gt_dir /kaggle/working/my_v2a_test_data/audio_mels_gt/ \
    #   --custom_empty_vid_path /kaggle/working/my_v2a_test_data/empty_video_data/empty_vid.npz \
    #   --outdir /kaggle/working/my_v2a_output/ \
    #   --sample_rate 16000 \
    #   --scale 3.0 # Esempio se vuoi usare CFG

    main()