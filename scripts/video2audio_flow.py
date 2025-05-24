import argparse, os, sys, glob
import pathlib
directory = pathlib.Path(os.getcwd())
print(directory)
sys.path.append(str(directory))
import torch
import numpy as np
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import random, math, librosa
from vocoder.bigvgan.models import VocoderBigVGAN
import soundfile
from pathlib import Path
from tqdm import tqdm


def load_model_from_config(config, ckpt=None, verbose=True):
    print(f"DEBUG: Istanzio modello da config: {config.model}")  # Aggiunto debug
    model = instantiate_from_config(config.model)
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")

        # Controlla se 'state_dict' è la chiave corretta o se i pesi sono al primo livello
        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
            print(
                f'---------------------------epoch : {pl_sd.get("epoch", "N/A")}, global step: {pl_sd.get("global_step", 0) // 1e3}k---------------------------')  # Usato .get per sicurezza
        else:
            sd = pl_sd  # A volte i checkpoint non hanno la chiave 'state_dict'
            print("DEBUG: Chiave 'state_dict' non trovata nel checkpoint, uso il checkpoint direttamente.")
            print(
                f'---------------------------epoch: N/A, global step: N/A (struttura checkpoint diversa)---------------------------')

        # --- INIZIO BLOCCO DI DEBUG E CORREZIONE CHIAVI ---
        print("\nDEBUG: Prime 5 chiavi del MODELLO (model.state_dict().keys()):")
        model_keys = list(model.state_dict().keys())
        for i, key in enumerate(model_keys[:5]):
            print(f"  {key}")

        print("\nDEBUG: Prime 5 chiavi del CHECKPOINT CARICATO (sd.keys()):")
        checkpoint_keys = list(sd.keys())
        for i, key in enumerate(checkpoint_keys[:5]):
            print(f"  {key}")

        # Logica di correzione potenziale:
        # Se le chiavi del modello iniziano con 'model.' e quelle del checkpoint no,
        # aggiungiamo 'model.' alle chiavi del checkpoint.

        # Controlliamo se dobbiamo aggiungere il prefisso 'model.'
        needs_prefix_addition = False
        if model_keys and checkpoint_keys:  # Assicurati che entrambe le liste non siano vuote
            if model_keys[0].startswith("model.") and not checkpoint_keys[0].startswith("model."):
                needs_prefix_addition = True
                print(
                    "\nINFO DEBUG: Rilevato che le chiavi del modello hanno 'model.' e quelle del checkpoint no. Tenterò di AGGIUNGERE il prefisso 'model.' alle chiavi del checkpoint.")
            elif not model_keys[0].startswith("model.") and checkpoint_keys[0].startswith("model."):
                print(
                    "\nINFO DEBUG: Rilevato che le chiavi del checkpoint hanno 'model.' e quelle del modello no. Tenterò di RIMUOVERE il prefisso 'model.' dalle chiavi del checkpoint.")
                # Questo è il caso che il tuo log originale suggeriva fosse in atto
                # ma i messaggi di missing/unexpected indicano il contrario per questo specifico checkpoint.
                new_sd = {}
                for k, v in sd.items():
                    if k.startswith("model."):
                        new_sd[k[len("model."):]] = v
                    else:
                        new_sd[k] = v  # Mantieni le chiavi che non hanno il prefisso
                sd = new_sd
                print("DEBUG: Prefisso 'model.' rimosso dalle chiavi del checkpoint (se presente).")
                # Ricontrolla le chiavi del checkpoint dopo la rimozione
                print("\nDEBUG: Prime 5 chiavi del CHECKPOINT dopo tentativo di rimozione prefisso:")
                checkpoint_keys_after_removal = list(sd.keys())
                for i, key in enumerate(checkpoint_keys_after_removal[:5]):
                    print(f"  {key}")

        if needs_prefix_addition:
            new_sd = {}
            for k, v in sd.items():
                new_sd[f"model.{k}"] = v
            sd = new_sd  # Sovrascrivi sd con le chiavi modificate
            print("DEBUG: Prefisso 'model.' AGGIUNTO alle chiavi del checkpoint.")
            # Ricontrolla le chiavi del checkpoint dopo l'aggiunta
            print("\nDEBUG: Prime 5 chiavi del CHECKPOINT dopo AGGIUNTA prefisso:")
            checkpoint_keys_after_addition = list(sd.keys())
            for i, key in enumerate(checkpoint_keys_after_addition[:5]):
                print(f"  {key}")

        # --- FINE BLOCCO DI DEBUG E CORREZIONE CHIAVI ---

        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print(
                "ATTENZIONE: Ancora chiavi mancanti nel modello dopo la correzione (missing keys):")  # Cambiato in ATTENZIONE
            print(m)
        if len(u) > 0 and verbose:
            print(
                "ATTENZIONE: Ancora chiavi inaspettate nel checkpoint dopo la correzione (unexpected keys):")  # Cambiato in ATTENZIONE
            print(u)

        if not m and not u:  # Se entrambe le liste sono vuote
            print("INFO: Tutte le chiavi del checkpoint sono state caricate con successo nel modello!")

    else:
        print(f"Note chat no ckpt is loaded !!!")

    model.cuda()
    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sample_rate",
        type=int,
        default="16000",
        help="sample rate of wav"
    )
    parser.add_argument(
        "--length",
        type=int,
        default=None,
        help="length of wav"
    )
    parser.add_argument(
        "--test-dataset",
        default="vggsound",
        help="test which dataset: vggsound/landscape/fsd50k"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2audio-samples"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=25,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=1.0, # if it's 1, only condition is taken into consideration
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        type=str,
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default="",
    )


    return parser.parse_args()


import os
import torch  # Assicurati che torch sia importato
import numpy as np  # Assicurati che numpy sia importato
from omegaconf import OmegaConf  # Assicurati che OmegaConf sia importato
from ldm.util import instantiate_from_config  # Assicurati che sia importato
from vocoder.bigvgan.models import VocoderBigVGAN  # Assicurati che sia importato
from pathlib import Path  # Assicurati che Path sia importato
import math  # Assicurati che math sia importato
from tqdm import tqdm  # Assicurati che tqdm sia importato
import soundfile  # Assicurati che soundfile sia importato


# Aggiungi anche l'import per load_model_from_config se è definito in un altro file
# from ..tuo_modulo import load_model_from_config # Esempio se in un altro file
# o assicurati che sia definito prima di main() se nello stesso file.
# Aggiungi import per parse_args se è definito in un altro file
# from ..tuo_modulo_args import parse_args # Esempio se in un altro file

# Se parse_args e load_model_from_config sono nello stesso file di video2audio_flow.py
# e definiti prima di main(), non servono import espliciti per loro qui.

def main():
    opt = parse_args()

    # === CONFIGURAZIONE PER TEST SINGOLO VIDEO ===
    PROJECT_ROOT_ON_KAGGLE = str(pathlib.Path(os.getcwd()))  # Usa il CWD rilevato all'inizio

    # ---!!! MODIFICA QUESTI PERCORSI CON I TUOI FILE DI TEST !!!---
    # 1. Path al file .npz delle FEATURE VIDEO del tuo video di test
    #    Questo file deve contenere un array sotto la chiave 'feat'.
    #    Esempio: "/kaggle/working/my_single_test_data/my_video_clip_features.npz"
    SINGLE_VIDEO_FEAT_PATH = "/kaggle/working/my_generated_cavp_features/chip_test.npz"  # <--- MODIFICA QUESTO!
    if not os.path.exists(SINGLE_VIDEO_FEAT_PATH):
        print(f"ERRORE: File feature video '{SINGLE_VIDEO_FEAT_PATH}' non trovato! Crea questo file prima.")
        return

    # 2. (Opzionale) Path al file _mel.npy dello SPETTROGRAMMA MEL ground truth del tuo video di test
    #    Esempio: "/kaggle/working/my_single_test_data/my_video_audio_mel.npy"
    SINGLE_SPEC_PATH = "/kaggle/working/my_single_test_data/chip_test_mel.npy"  # <--- MODIFICA QUESTO (o imposta a None)
    if SINGLE_SPEC_PATH and not os.path.exists(SINGLE_SPEC_PATH):
        print(f"ATTENZIONE: File spettrogramma Mel GT '{SINGLE_SPEC_PATH}' non trovato. Procedo senza GT audio.")
        SINGLE_SPEC_PATH = None

        # 3. Nome base per i file di output (senza estensione)
    OUTPUT_NAME_BASE = Path(SINGLE_VIDEO_FEAT_PATH).stem
    if OUTPUT_NAME_BASE.endswith("_features") or OUTPUT_NAME_BASE.endswith("_feat"):  # Pulisce nomi comuni
        OUTPUT_NAME_BASE = OUTPUT_NAME_BASE.rsplit('_', 1)[0]
    print(f"DEBUG: Nome base per output: {OUTPUT_NAME_BASE}")

    # 4. Path per il file unconditional 'empty_vid.npz'
    UNCONDITIONAL_FEAT_PATH_ON_KAGGLE = os.path.join(PROJECT_ROOT_ON_KAGGLE, "useful_ckpts", "empty_vid.npz")
    if not os.path.exists(UNCONDITIONAL_FEAT_PATH_ON_KAGGLE) and opt.scale != 1:
        print(
            f"ERRORE: File unconditional '{UNCONDITIONAL_FEAT_PATH_ON_KAGGLE}' non trovato e opt.scale ({opt.scale}) != 1!")
        return
    # === FINE CONFIGURAZIONE PER TEST SINGOLO VIDEO ===

    config = OmegaConf.load(opt.base)
    model = load_model_from_config(config, opt.resume)  # Usa la tua funzione originale

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    print(f"DEBUG: Modello spostato su {device}")

    os.makedirs(opt.outdir, exist_ok=True)
    print(f"DEBUG: Directory di output assicurata: {opt.outdir}")

    vocoder_ckpt_path = config['lightning']['callbacks']['image_logger']['params']['vocoder_cfg']['params'][
        'ckpt_vocoder']
    if not os.path.isabs(vocoder_ckpt_path) and not vocoder_ckpt_path.startswith(
            "/kaggle/input"):  # Considera anche /kaggle/working
        potential_path_working = os.path.join(PROJECT_ROOT_ON_KAGGLE, vocoder_ckpt_path)
        if os.path.exists(potential_path_working):
            vocoder_ckpt_path = potential_path_working
    print(f"DEBUG: Uso checkpoint vocoder: {vocoder_ckpt_path}")
    if not os.path.exists(vocoder_ckpt_path):
        print(f"ERRORE CRITICO: Checkpoint del Vocoder non trovato in '{vocoder_ckpt_path}'. Verifica il config YAML.")
        return
    vocoder = VocoderBigVGAN(vocoder_ckpt_path, device)

    sr = opt.sample_rate
    # Estrai i parametri del dataset dalla config
    dataset_params_key = 'test' if 'test' in config.data.params else 'train'  # Preferisci test se esiste
    dataset_cfg_path = config.data.params.get(dataset_params_key, {}).get('params', {}).get('dataset_cfg', {})
    if not dataset_cfg_path:  # Fallback se la struttura è diversa o dataset_cfg è al livello superiore
        dataset_cfg_path = config.data.params.get('dataset_cfg', {})

    duration_config = dataset_cfg_path.get('duration', 10)  # Default a 10 se non trovato
    truncate_duration_config = dataset_cfg_path.get('truncate', duration_config)
    fps_config = dataset_cfg_path.get('fps', 21.5)
    hop_len_config = dataset_cfg_path.get('hop_len', 256)

    # truncate_frame è la lunghezza della finestra di processamento in frame video
    truncate_frame_per_window = int(fps_config * truncate_duration_config)
    print(
        f"DEBUG: sr={sr}, duration_cfg={duration_config}, truncate_duration_cfg={truncate_duration_config}, fps_cfg={fps_config}, hop_len_cfg={hop_len_config}, truncate_frame_per_window={truncate_frame_per_window}")

    uc = None
    if opt.scale != 1:
        unconditional_data = np.load(UNCONDITIONAL_FEAT_PATH_ON_KAGGLE)
        if 'feat' in unconditional_data:
            unconditional_np = unconditional_data['feat'].astype(np.float32)
        elif 'arr_0' in unconditional_data:
            unconditional_np = unconditional_data['arr_0'].astype(np.float32)
        else:
            print(f"ERRORE: Chiave 'feat' o 'arr_0' non trovata in {UNCONDITIONAL_FEAT_PATH_ON_KAGGLE}")
            return

        if unconditional_np.shape[0] < truncate_frame_per_window:
            unconditional_np = np.tile(unconditional_np,
                                       (math.ceil(truncate_frame_per_window / unconditional_np.shape[0]), 1))
        unconditional_np = unconditional_np[:truncate_frame_per_window]

        unconditional = torch.from_numpy(unconditional_np).unsqueeze(0).to(device)
        uc = model.get_learned_conditioning(unconditional)
        print(f"DEBUG: UC (unconditional context) creato con shape: {uc.shape if uc is not None else 'None'}")

    try:
        video_feat_data = np.load(SINGLE_VIDEO_FEAT_PATH)
        if 'feat' in video_feat_data:
            video_feat_np = video_feat_data['feat'].astype(np.float32)
        elif 'arr_0' in video_feat_data:
            video_feat_np = video_feat_data['arr_0'].astype(np.float32)
        else:
            video_feat_np = np.load(SINGLE_VIDEO_FEAT_PATH).astype(np.float32)
    except Exception as e:
        print(f"ERRORE: Impossibile caricare feature video da '{SINGLE_VIDEO_FEAT_PATH}': {e}")
        return
    video_feat = torch.from_numpy(video_feat_np).unsqueeze(0).to(device)
    print(f"DEBUG: Feature video caricate. Shape: {video_feat.shape}. Dtype: {video_feat.dtype}")

    spec_raw = None  # Per ground truth
    if SINGLE_SPEC_PATH:
        try:
            spec_raw_np = np.load(SINGLE_SPEC_PATH).astype(np.float32)
            spec_raw = torch.from_numpy(spec_raw_np).unsqueeze(0).to(device)
            print(f"DEBUG: Spettrogramma Mel GT caricato. Shape: {spec_raw.shape}. Dtype: {spec_raw.dtype}")
        except Exception as e:
            print(f"ATTENZIONE: Impossibile caricare Mel GT da '{SINGLE_SPEC_PATH}': {e}. Genero senza GT audio.")

    gt_mel_list_for_concat, generated_mel_list_for_concat = [], []
    num_total_video_frames = video_feat.shape[1]

    if truncate_frame_per_window <= 0:
        print(
            f"ATTENZIONE: truncate_frame_per_window ({truncate_frame_per_window}) non valido. Uso l'intera lunghezza del video ({num_total_video_frames} frames) come una singola finestra.")
        effective_truncate_frame = num_total_video_frames
    else:
        effective_truncate_frame = truncate_frame_per_window

    window_num = math.ceil(num_total_video_frames / effective_truncate_frame)
    print(
        f"DEBUG: Video frames totali: {num_total_video_frames}, processando in finestre di {effective_truncate_frame} frames, #finestre: {window_num}")

    output_mel_shape_per_window = None  # Forma del Mel generato per finestra
    if opt.length is not None:  # opt.length è in frame Mel per L'INTERO audio, non per finestra
        # Per ora, se opt.length è specificato, lo useremo come target per l'output finale,
        # e il campionamento per finestra cercherà di generare la sua porzione.
        # Questo potrebbe richiedere una logica più complessa per la lunghezza per finestra se opt.length < lunghezza totale attesa.
        # La classe Model originale usa opt.length per la forma del campionatore.
        # Assumiamo che `model.mel_dim` esista e sia il numero di bin Mel.
        if not hasattr(model, 'mel_dim'):
            print("ERRORE: model.mel_dim non trovato. Necessario per specificare la forma con opt.length.")
            # Tenta di dedurlo dal primo stadio o dalla config, se possibile
            try:
                model.mel_dim = config.model.params.first_stage_config.params.ddconfig.get('in_channels',
                                                                                           config.model.params.unet_config.params.get(
                                                                                               'in_channels', 80))
                print(f"DEBUG: model.mel_dim dedotto/impostato a {model.mel_dim}")
            except AttributeError:
                print("ERRORE: Impossibile dedurre model.mel_dim dalla configurazione.")
                return

        output_mel_shape_overall = (1, model.mel_dim, opt.length)
        print(
            f"DEBUG: Lunghezza output Mel target specificata (opt.length): {opt.length} frames. Shape overall: {output_mel_shape_overall}")
        # La logica di ntk_factor per lunghezze diverse dal training è complessa e dipende dal DiT.
        # Lasciamo che il campionatore gestisca la forma per finestra se possibile.
        # Se opt.length si riferisce all'output totale, ogni finestra genererà una porzione.
        # La forma passata al campionatore sarà quindi per la singola finestra.

    for i_window in tqdm(range(window_num), desc=f"Finestra ({OUTPUT_NAME_BASE})"):
        start_vframe = i_window * effective_truncate_frame
        end_vframe = min((i_window + 1) * effective_truncate_frame, num_total_video_frames)

        if start_vframe >= end_vframe: continue

        current_video_segment_for_cond = video_feat[:, start_vframe:end_vframe]
        current_video_segment_actual_len_frames = current_video_segment_for_cond.shape[1]

        # Padding del segmento video per CFG se necessario (se è l'ultima finestra ed è più corta)
        # e se uc (unconditional context) ha una lunghezza fissa (effective_truncate_frame)
        c_input_segment = current_video_segment_for_cond
        if opt.scale != 1 and uc is not None and current_video_segment_for_cond.shape[1] < effective_truncate_frame:
            padding_size = effective_truncate_frame - current_video_segment_for_cond.shape[1]
            padding = torch.zeros(
                (current_video_segment_for_cond.shape[0], padding_size, current_video_segment_for_cond.shape[2]),
                device=device, dtype=current_video_segment_for_cond.dtype)
            c_input_segment = torch.cat([current_video_segment_for_cond, padding], dim=1)
            # print(f"  DEBUG Finestra {i_window}: Segmento video paddato da {current_video_segment_actual_len_frames} a {c_input_segment.shape[1]} per CFG.")

        c = model.get_learned_conditioning(c_input_segment)
        # print(f"  DEBUG Finestra {i_window}: Video frames {start_vframe}-{end_vframe-1}. Cond c shape: {c.shape if hasattr(c, 'shape') else type(c)}")

        # Determina la forma dell'output Mel per questa finestra
        # La lunghezza Mel dovrebbe corrispondere alla lunghezza *effettiva* del segmento video (non paddato)
        mel_len_for_this_segment_actual = int(
            current_video_segment_actual_len_frames / fps_config * sr / hop_len_config)

        # Se opt.length è specificato, dobbiamo decidere come dividerlo tra le finestre.
        # Per semplicità, se opt.length è dato, ogni finestra genererà Mel di lunghezza fissa (opt.length / window_num)
        # ma questo può portare a disallineamenti se le finestre video non sono tutte della stessa lunghezza effettiva.
        # Un approccio migliore è generare Mel proporzionale alla lunghezza video effettiva della finestra.
        if opt.length is not None and window_num > 0:
            # Questo è un calcolo approssimativo della lunghezza Mel per questa finestra se opt.length è per il totale
            # Potrebbe essere meglio lasciare che 'current_shape_for_sampler' sia basato su mel_len_for_this_segment_actual
            # e poi tagliare/paddare il risultato finale concatenato a opt.length.
            # Per ora, usiamo mel_len_for_this_segment_actual per la generazione per finestra.
            current_shape_for_sampler = (1, model.mel_dim, mel_len_for_this_segment_actual)
        else:
            current_shape_for_sampler = (1, model.mel_dim, mel_len_for_this_segment_actual)

        # print(f"  DEBUG Finestra {i_window}: Shape output Mel per campionatore: {current_shape_for_sampler}")
        if current_shape_for_sampler[2] <= 0:  # Salta se la lunghezza Mel calcolata è zero o negativa
            print(
                f"  ATTENZIONE Finestra {i_window}: Lunghezza Mel calcolata non positiva ({current_shape_for_sampler[2]}). Salto questa finestra.")
            continue

        if opt.scale == 1 or uc is None:
            if uc is None and opt.scale != 1: print(
                f"  ATTENZIONE Finestra {i_window}: uc non disponibile, fallback a no CFG.")
            sample_latents, _ = model.sample(c, 1, timesteps=opt.ddim_steps, shape=current_shape_for_sampler)
        else:
            sample_latents, _ = model.sample_cfg(c, opt.scale, uc, 1, timesteps=opt.ddim_steps,
                                                 shape=current_shape_for_sampler)

        generated_mel_this_window = model.decode_first_stage(sample_latents)
        generated_mel_list_for_concat.append(generated_mel_this_window)

        if spec_raw is not None:
            start_spec_frame = int(start_vframe / fps_config * sr / hop_len_config)
            # La lunghezza del GT Mel deve corrispondere alla lunghezza del Mel generato per questa finestra
            end_spec_frame = start_spec_frame + generated_mel_this_window.shape[2]

            if start_spec_frame < spec_raw.shape[2]:
                current_gt_mel_segment = spec_raw[:, :, start_spec_frame: min(end_spec_frame, spec_raw.shape[2])]
                if current_gt_mel_segment.shape[2] > 0:
                    # Allinea/Padda il GT Mel per corrispondere alla lunghezza del Mel generato
                    if current_gt_mel_segment.shape[2] < generated_mel_this_window.shape[2]:
                        padding_gt = torch.zeros_like(generated_mel_this_window)
                        padding_gt[:, :, :current_gt_mel_segment.shape[2]] = current_gt_mel_segment
                        current_gt_mel_segment = padding_gt
                    elif current_gt_mel_segment.shape[2] > generated_mel_this_window.shape[2]:
                        current_gt_mel_segment = current_gt_mel_segment[:, :, :generated_mel_this_window.shape[2]]
                    gt_mel_list_for_concat.append(current_gt_mel_segment)
                else:
                    gt_mel_list_for_concat.append(torch.zeros_like(generated_mel_this_window))
            else:
                gt_mel_list_for_concat.append(torch.zeros_like(generated_mel_this_window))

    # Concatena e salva
    if generated_mel_list_for_concat:
        syn_mel_final = torch.cat([mel.cpu().detach() for mel in generated_mel_list_for_concat], dim=2)

        # Se opt.length è specificato, taglia/padda il Mel finale a quella lunghezza
        if opt.length is not None and syn_mel_final.shape[2] != opt.length:
            if syn_mel_final.shape[2] > opt.length:
                syn_mel_final = syn_mel_final[:, :, :opt.length]
                print(f"DEBUG: Melodia sintetizzata tagliata a opt.length: {opt.length}. Shape: {syn_mel_final.shape}")
            else:  # syn_mel_final.shape[2] < opt.length
                padding_final = torch.zeros(
                    (syn_mel_final.shape[0], syn_mel_final.shape[1], opt.length - syn_mel_final.shape[2]),
                    dtype=syn_mel_final.dtype)
                syn_mel_final = torch.cat([syn_mel_final, padding_final], dim=2)
                print(f"DEBUG: Melodia sintetizzata paddata a opt.length: {opt.length}. Shape: {syn_mel_final.shape}")

        print(f"DEBUG: Melodia sintetizzata finale da salvare. Shape: {syn_mel_final.shape}")
        for idx_batch in range(syn_mel_final.shape[0]):  # Dovrebbe essere 1
            ddim_wav = vocoder.vocode(syn_mel_final[idx_batch])  # Input a vocoder: [mel_bins, time_frames]
            wav_path = os.path.join(opt.outdir, OUTPUT_NAME_BASE + f'_{idx_batch}_generated.wav')
            soundfile.write(wav_path, ddim_wav, sr)  # Usa sr (opt.sample_rate) per salvare
            print(f"Audio sintetizzato salvato in: {wav_path}")
    else:
        print("ERRORE: Nessun Mel sintetizzato generato (generated_mel_list_for_concat è vuota).")

    if gt_mel_list_for_concat and spec_raw is not None:
        valid_gt_mels = [mel.cpu().detach() for mel in gt_mel_list_for_concat if
                         mel.nelement() > 0 and mel.shape[2] > 0]
        if valid_gt_mels:
            if len(set(m.shape[1] for m in valid_gt_mels)) <= 1:  # Tutti hanno lo stesso numero di bin Mel
                gt_mel_final = torch.cat(valid_gt_mels, dim=2)

                # Se opt.length è specificato, taglia/padda anche il GT Mel finale
                if opt.length is not None and gt_mel_final.shape[2] != opt.length:
                    if gt_mel_final.shape[2] > opt.length:
                        gt_mel_final = gt_mel_final[:, :, :opt.length]
                    else:
                        padding_final_gt = torch.zeros(
                            (gt_mel_final.shape[0], gt_mel_final.shape[1], opt.length - gt_mel_final.shape[2]),
                            dtype=gt_mel_final.dtype)
                        gt_mel_final = torch.cat([gt_mel_final, padding_final_gt], dim=2)

                print(f"DEBUG: Melodia GT finale da salvare. Shape: {gt_mel_final.shape}")
                for idx_batch in range(gt_mel_final.shape[0]):
                    wav_gt = vocoder.vocode(gt_mel_final[idx_batch])
                    wav_gt_path = os.path.join(opt.outdir, OUTPUT_NAME_BASE + f'_{idx_batch}_gt.wav')
                    soundfile.write(wav_gt_path, wav_gt, sr)
                    print(f"Audio GT salvato in: {wav_gt_path}")
            else:
                print(
                    "ATTENZIONE: Non tutti i segmenti GT Mel validi hanno lo stesso numero di canali Mel. Impossibile concatenare GT.")
        else:
            print("ATTENZIONE: Nessun segmento GT Mel valido da concatenare.")

    print(f"Processo completato. Output (se generato) in: \n{opt.outdir}")


if __name__ == "__main__":
    # Assicurati che le funzioni necessarie siano definite o importate
    # (parse_args, load_model_from_config, etc.)
    main()

