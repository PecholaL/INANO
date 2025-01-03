"""get speaker embedding via MAIN-VC.Enc_S
"""

import os
import torch
import torch.nn.functional as F
import yaml
import pickle
from argparse import ArgumentParser
from scipy.io.wavfile import write
from MAIN_VC.main_vc import MAINVC
from MAIN_VC.tools import *
from MAIN_VC.utils import *
from MAIN_VC.hyper_params import HyperParameters as hp


class Inferencer(object):
    def __init__(self, config, args):
        self.config = config
        self.args = args

        self.build_model()
        self.load_model()
        self.load_vocoder()

        # mean, stdev
        with open(self.args.attr, "rb") as f:
            self.attr = pickle.load(f)

    def build_model(self):
        self.model = cc(MAINVC(self.config))
        self.model.eval()
        return

    def load_model(self):
        print(f"[MAIN-VC]load model from {self.args.model}")
        self.model.load_state_dict(torch.load(f"{self.args.model}"))
        return

    def load_vocoder(self):
        print("[MAIN-VC]load vocoder from {self.args.vocoder}")
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocoder = torch.jit.load(f"{self.args.vocoder}").to("cpu").eval()

    def toWav(self, mel):
        print(f"convert mel-spect shapes {mel.shape} into wav")
        with torch.no_grad():
            wav = self.vocoder.generate([torch.from_numpy(mel).float()])[0]
            wav = wav.cpu().numpy()
            print("generate wav")
            return wav

    def utt_make_frames(self, x):
        frame_size = self.config["data_loader"]["frame_size"]
        # x.shape (n_frames, n_mels)
        remains = x.size(0) % frame_size
        if remains != 0:
            x = F.pad(x, (0, remains))
        out = x.view(1, x.size(0) // frame_size, x.size(1) * frame_size).transpose(1, 2)
        # out.shape ((B)1, (C)n_mels, (W)n_frames)
        return out

    def inference_one_utterance(self, x, x_cond):
        x = self.utt_make_frames(x)  # (batch_size(1), n_mels, n_frames)
        x_cond = self.utt_make_frames(x_cond)  # (batch_size(1), n_mels, n_frames)
        dec = self.model.inference(x, x_cond)  # (batch_size(1), n_mels, n_frames)
        dec = dec.transpose(1, 2).squeeze(0)  # (n_frames, n_mels)
        dec = dec.detach().cpu().numpy()
        dec = self.denormalize(dec)
        wav_data = self.toWav(dec)
        return wav_data, dec

    def denormalize(self, x):
        m, s = self.attr["mean"], self.attr["std"]
        ret = x * s + m
        return ret

    def normalize(self, x):
        m, s = self.attr["mean"], self.attr["std"]
        ret = (x - m) / s
        return ret

    def inference_from_path(self):
        src_wav = load_wav(self.args.source, hp.sr)
        tar_wav = load_wav(self.args.target, hp.sr)
        src_mel = log_mel_spectrogram(
            src_wav,
            hp.preemph,
            hp.sr,
            hp.n_mels,
            hp.n_fft,
            hp.hop_len,
            hp.win_len,
            hp.f_min,
        )
        tar_mel = log_mel_spectrogram(
            tar_wav,
            hp.preemph,
            hp.sr,
            hp.n_mels,
            hp.n_fft,
            hp.hop_len,
            hp.win_len,
            hp.f_min,
        )
        src_mel = torch.from_numpy(self.normalize(src_mel)).float().cuda()
        tar_mel = torch.from_numpy(self.normalize(tar_mel)).float().cuda()
        conv_wav, _ = self.inference_one_utterance(src_mel, tar_mel)
        # Linux
        # src_info = self.args.source.split("/")[-1][:-4]
        # tar_info = self.args.target.split("/")[-1][:-4]
        # Windows
        src_info = self.args.source.split("\\")[-1][:-4]
        tar_info = self.args.target.split("\\")[-1][:-4]
        write(
            f"{self.args.output}/{src_info}_{tar_info}.wav",
            rate=self.args.sample_rate,
            data=conv_wav,
        )
        return
    
    def get_spk_emb(self, path):
        orig_wav = load_wav(path, hp.sr)
        orig_mel = log_mel_spectrogram(
            orig_wav,
            hp.preemph,
            hp.sr,
            hp.n_mels,
            hp.n_fft,
            hp.hop_len,
            hp.win_len,
            hp.f_min,
        )
        orig_mel = torch.from_numpy(self.normalize(orig_mel)).float().cuda()
        orig_mel = self.utt_make_frames(orig_mel)
        spk_emb = self.model.get_speaker_embedding(orig_mel).squeeze().float().detach().cpu().numpy()
        return spk_emb
    
    def get_all_spk_embs(self, root_dir, dataset_save_path):
        spk_emb_list = []
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.wav'):
                    wav_path = os.path.join(subdir, file)
                    spk_emb = self.get_spk_emb(wav_path)
                    spk_emb_list.append(spk_emb)

        print(f"get {len(spk_emb_list)} speaker embeddings, saving...")
        np.save(dataset_save_path, np.array(spk_emb_list))
        print(f"all spk_embs are saved to {dataset_save_path}")



"""test
"""
if __name__ == "__main__":
    parser = ArgumentParser()
    # Linux
    # parser.add_argument(
    #     "--attr",
    #     "-a",
    #     help="attr file path",
    #     default="/Users/pecholalee/Coding/VC/mainVc_data/attr.pkl",
    # )
    # parser.add_argument(
    #     "--config",
    #     "-c",
    #     help="config file path",
    #     default="/Users/pecholalee/Coding/VC/MAIN-VC/config.yaml",
    # )
    # parser.add_argument(
    #     "--model",
    #     "-m",
    #     help="model path",
    #     default="/Users/pecholalee/Coding/VC/mainVc_data/save/mainVcModel.ckpt",
    # )
    # parser.add_argument(
    #     "--vocoder",
    #     "-v",
    #     help="vocoder path",
    #     default="/Users/pecholalee/Coding/VC/mainVc_data/vocoder/vocoder.pt",
    # )

    # Windows
    parser.add_argument(
        "--attr",
        "-a",
        help="attr file path",
        default="c:\\Users\\leeklll\\Documents\\DL\\mainVc_data\\attr.pkl",
    )
    parser.add_argument(
        "--config",
        "-c",
        help="config file path",
        default="c:\\Users\\leeklll\\Documents\\DL\\MAIN-VC\\config.yaml",
    )
    parser.add_argument(
        "--model",
        "-m",
        help="model path",
        default="c:\\Users\\leeklll\\Documents\\DL\\mainVc_data\\save\\mainVcModel.ckpt",
    )
    parser.add_argument(
        "--vocoder",
        "-v",
        help="vocoder path",
        default="c:\\Users\\leeklll\\Documents\\DL\\mainVc_data\\vocoder\\vocoder.pt",
    )
    parser.add_argument("-source", "-s", help="source wav path")
    parser.add_argument("-target", "-t", help="target wav path")
    parser.add_argument("-output", "-o", help="output wav path")
    parser.add_argument(
        "--sample_rate", "-r", help="sample rate", default=16000, type=int
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    inferencer = Inferencer(config=config, args=args)
    # inferencer.inference_from_path()

    path = "c:\\Users\\leeklll\\Documents\\DL\\datasets\\archive\\VCTK-Corpus\\VCTK-Corpus\\wav48\\p225\\p225_001.wav"
    spk_emb = inferencer.get_spk_emb(path)
    print(spk_emb.shape)

    speech_corpus_root = "c:\\Users\\leeklll\\Documents\\DL\\datasets\\archive\\VCTK-Corpus\\VCTK-Corpus\\wav48"
    spk_emb_dataset_path = "c:\\Users\\leeklll\\Documents\\DL\\inano_data\\spk_emb.npy"
    inferencer.get_all_spk_embs(speech_corpus_root, spk_emb_dataset_path)