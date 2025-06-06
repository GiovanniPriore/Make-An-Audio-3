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
import pandas as pd
from tqdm import tqdm
import preprocess.n2s_by_openai as n2s
from vocoder.bigvgan.models import VocoderBigVGAN
import soundfile
import torchaudio, math
def load_model_from_config(config, ckpt = None, verbose=True):
    model = instantiate_from_config(config.model)
    if ckpt:
        print(f"Loading model from {ckpt}")
        # Riga originale:
        # pl_sd = torch.load(ckpt, map_location="cpu")
        # Riga Modificata:
        pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = pl_sd["state_dict"]
        
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)
    else:
        print(f"Note chat no ckpt is loaded !!!")

    model.cuda()
    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        default="A large truck driving by as an emergency siren wails and truck horn honks",
        help="the prompt to generate"
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default="16000",
        help="sample rate of wav"
    )
    parser.add_argument(
        "--test-dataset",
        default="audio",
        help="test which dataset: testset"
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
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )

    parser.add_argument(
        "--H",
        type=int,
        default=20, # keep fix
        help="latent height, in pixel space",
    )

    parser.add_argument(
        "--W",
        type=int,
        default=312, # keep fix
        help="latent width, in pixel space",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for the given prompt",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=5.0, # if it's 1, only condition is taken into consideration
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
    parser.add_argument(
        "--vocoder-ckpt",
        type=str,
        help="paths to vocoder checkpoint",
        default='vocoder/logs/audioset',
    )

    return parser.parse_args()

class GenSamples:
    def __init__(self,opt, model,outpath,config, vocoder = None,save_mel = True,save_wav = True) -> None:
        self.opt = opt
        self.model = model
        self.outpath = outpath
        if save_wav:
            assert vocoder is not None
            self.vocoder = vocoder
        self.save_mel = save_mel
        self.save_wav = save_wav
        self.channel_dim = self.model.channels
        self.config = config
    
    def gen_test_sample(self,prompt, mel_name = None,wav_name = None, gt=None, video=None):# prompt is {'ori_caption':’xxx‘,'struct_caption':'xxx'}
        uc = None
        record_dicts = []
        if self.opt.scale != 1.0:
            try: # audiocaps
                uc = self.model.get_learned_conditioning({'ori_caption': "",'struct_caption': ""})
            except: # audioset, music
                uc = self.model.get_learned_conditioning(prompt['ori_caption'])
        for n in range(self.opt.n_iter):# trange(self.opt.n_iter, desc="Sampling"):

            try: # audiocaps
                c = self.model.get_learned_conditioning(prompt) # shape:[1,77,1280],
            except: # audioset
                c = self.model.get_learned_conditioning(prompt['ori_caption'])

            if self.channel_dim>0:
                shape = [self.channel_dim, self.opt.H, self.opt.W]  # (z_dim, 80//2^x, 848//2^x)
            else:
                shape = [1, self.opt.H, self.opt.W]

            x0 = torch.randn(shape, device=self.model.device)

            if self.opt.scale == 1: # w/o cfg
                sample, _ = self.model.sample(c, 1, timesteps=self.opt.ddim_steps, x_latent=x0)
            else:  # cfg
                sample, _ = self.model.sample_cfg(c, self.opt.scale, uc, 1, timesteps=self.opt.ddim_steps, x_latent=x0)

            x_samples_ddim = self.model.decode_first_stage(sample)

            for idx,spec in enumerate(x_samples_ddim):
                spec = spec.squeeze(0).cpu().numpy()
                record_dict = {'caption':prompt['ori_caption'][0]}
                if self.save_mel:
                    mel_path = os.path.join(self.outpath,mel_name+f'_{idx}.npy')
                    np.save(mel_path,spec)
                    record_dict['mel_path'] = mel_path
                if self.save_wav:
                    wav = self.vocoder.vocode(spec)
                    wav_path = os.path.join(self.outpath,wav_name+f'_{idx}.wav')
                    soundfile.write(wav_path, wav, self.opt.sample_rate)
                    record_dict['audio_path'] = wav_path
                record_dicts.append(record_dict)


        # --- MODIFICA QUI ---
        # Processa e salva il ground truth solo se gt è fornito
        if gt is not None:
            print(f"DEBUG: Processazione del ground truth (gt) fornito.")
            # Assicurati che gt sia un tensore o array numpy prima di passarlo al vocoder
            # Se gt è un path a un file, dovresti caricarlo prima.
            # Assumendo che gt sia già uno spettrogramma (come per il dataset di test)
            try:
                wav_gt = self.vocoder.vocode(gt) # gt dovrebbe essere lo spettrogramma
                # Assicurati che wav_name sia definito anche se mel_name non lo è
                gt_wav_filename_base = wav_name if wav_name else "ground_truth_audio"
                wav_path_gt = os.path.join(self.outpath, gt_wav_filename_base + f'_gt.wav')
                soundfile.write(wav_path_gt, wav_gt, 16000) # La sample rate per GT è 16000 nel codice originale. Adatta se necessario.
                print(f"Audio Ground Truth salvato in: {wav_path_gt}")
            except Exception as e_gt_vocode:
                print(f"WARN: Errore durante la processazione del ground truth (gt): {e_gt_vocode}")
                print(f"      gt era: {type(gt)}")

        else:
            print("DEBUG: Nessun ground truth (gt) fornito, salto la sua processazione.")
        # --- FINE MODIFICA ---

        return record_dicts


def main():
    opt = parse_args()

    config = OmegaConf.load(opt.base)
    # print("-------quick debug no load ckpt---------")
    # model = instantiate_from_config(config['model'])# for quick debug
    model = load_model_from_config(config, opt.resume)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)


    os.makedirs(opt.outdir, exist_ok=True)
    vocoder = VocoderBigVGAN(opt.vocoder_ckpt,device)



    generator = GenSamples(opt, model,opt.outdir,config, vocoder,save_mel = False,save_wav = True)
    csv_dicts = []
    
    with torch.no_grad():
        with model.ema_scope():
            if opt.test_dataset == 'testset':
                test_dataset = instantiate_from_config(config['test_dataset'])
                video = None

                print(f"Dataset: {type(test_dataset)} LEN: {len(test_dataset)}")
                for item in tqdm(test_dataset):
                    prompt, f_name, gt = item['caption'], item['f_name'], item['image']
                    vname_num_split_index = f_name.rfind('_')  # file_names[b]:video_name+'_'+num
                    v_n, num = f_name[:vname_num_split_index], f_name[vname_num_split_index + 1:]
                    mel_name = f'{v_n}_sample_{num}'
                    wav_name = f'{v_n}_sample_{num}'
                    # write_gt_wav(v_n,opt.test_dataset2,opt.outdir,opt.sample_rate)
                    csv_dicts.extend(generator.gen_test_sample(prompt, mel_name=mel_name, wav_name=wav_name, gt=gt, video=video))

                    df = pd.DataFrame.from_dict(csv_dicts)
                    df.to_csv(os.path.join(opt.outdir,'result.csv'),sep='\t',index=False)

            elif opt.test_dataset == 'structure':
                ori_caption = opt.prompt
                struct_caption = n2s.get_struct(ori_caption)
                print(f"The structed caption by Chatgpt is : {struct_caption}")
                wav_name = f'{ori_caption.strip().replace(" ", "-")}'
                prompt = {'ori_caption':[ori_caption],'struct_caption':[struct_caption]}
                generator.gen_test_sample(prompt, wav_name=wav_name)

            else:
                ori_caption = opt.prompt
                wav_name = f'{ori_caption.strip().replace(" ", "-")}'
                # Modifica: Aggiungi struct_caption
                # Opzione A.1: struct_caption uguale a ori_caption
                struct_caption_for_prompt = ori_caption
                # Opzione A.2: struct_caption stringa vuota se non la usi
                # struct_caption_for_prompt = "" 
                prompt = {
                    'ori_caption': [ori_caption],
                    'struct_caption': [struct_caption_for_prompt]  # Assicurati sia una lista di stringhe
                }
                print(f"DEBUG: Prompt preparato per gen_test_sample: {prompt}")  # Per debug
                generator.gen_test_sample(prompt, wav_name=wav_name)

    print(f"Your samples are ready and waiting four you here: \n{opt.outdir} \nEnjoy.")

if __name__ == "__main__":
    main()

