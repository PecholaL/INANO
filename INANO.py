""" INANO
    * Content encoder (EncC) and decoder (Dec) are taken from VC model (AdaIN-VC/MAIN-VC)
    * Generator (G) is based on Glow
    ** EncC and Dec are trained via the VC model's reconstruction process
    ** G is trained on the speaker encoder dataset output by the speaker encoder of the VC model
    ! Note: All the modules mentioned above are trained and INANO only emsembles the pre-trained modules.
    ! i.e. INANO does NOT need additional training.
    """

import yaml
from MAIN_VC.main_vc import MAINVC
from Gano import getGLOW


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
from Gano import getGLOW


class INANO(object):
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
        self.mainvc = cc(MAINVC(self.config))
        self.mainvc.eval()
        self.gano = cc(getGLOW())
        self.gano.eval()
        return

    def load_model(self):
        print(f"[INANO]load model from {self.args.mainvc} and {self.args.gano}")
        self.mainvc.load_state_dict(torch.load(f"{self.args.mainvc}"))
        self.gano.load_state_dict(torch.load(f"{self.args.gano}"))
        return

    def load_vocoder(self):
        print(f"[INANO]load vocoder from {self.args.vocoder}")
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocoder = torch.jit.load(f"{self.args.vocoder}").to("cpu").eval()

    def toWav(self, mel):
        print(f"[INANO]convert mel-spect shapes {mel.shape} into wav")
        with torch.no_grad():
            wav = self.vocoder.generate([torch.from_numpy(mel).float()])[0]
            wav = wav.cpu().numpy()
            print("[INANO]wav generated")
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

    def anonymize_utterance(self, x):
        x = self.utt_make_frames(x)  # original speech mel (batch_size(1), n_mels, n_frames)
        # generate ano spk embedding
        z = torch.randn(1, 64).to("cuda")
        with torch.no_grad():
            ano_spk_emb, _ = self.gano.inverse(z)
            ano_spk_emb = ano_spk_emb.reshape(1,64)
        # dec = self.mainvc.inference(x, x_cond)  # (batch_size(1), n_mels, n_frames)
        dec = self.mainvc.inference_for_inano(x, ano_spk_emb)
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

    def anonymize_from_path(self):
        orig_wav = load_wav(self.args.orig, hp.sr)
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
        ano_wav, _ = self.anonymize_utterance(orig_mel)
        # Linux
        # orig_info = self.args.orig.split("/")[-1][:-4]
        # Windows
        orig_info = self.args.orig.split("\\")[-1][:-4]
        write(
            f"{self.args.output}/{orig_info}_ano.wav",
            rate=self.args.sample_rate,
            data=ano_wav,
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
        spk_emb = self.mainvc.get_speaker_embedding(orig_mel).squeeze().float().detach().cpu().numpy()
        return spk_emb


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
        "--mainvc",
        "-m",
        help="mainvc model path",
        default="c:\\Users\\leeklll\\Documents\\DL\\mainVc_data\\save\\mainVcModel.ckpt",
    )
    parser.add_argument(
        "--gano",
        "-g",
        help="gano model path",
        default="c:\\Users\\leeklll\\Documents\\DL\\inano_data\\save\\gano.ckpt",
    )
    parser.add_argument(
        "--vocoder",
        "-v",
        help="vocoder path",
        default="c:\\Users\\leeklll\\Documents\\DL\\mainVc_data\\vocoder\\vocoder.pt",
    )
    parser.add_argument("-orig", "-s", help="original wav path")
    parser.add_argument("-output", "-o", help="output wav path")
    parser.add_argument(
        "--sample_rate", "-r", help="sample rate", default=16000, type=int
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    inano = INANO(config=config, args=args)
    inano.anonymize_from_path()

    # path = "c:\\Users\\leeklll\\Documents\\DL\\datasets\\archive\\VCTK-Corpus\\VCTK-Corpus\\wav48\\p225\\p225_001.wav"
    # spk_emb = inano.get_spk_emb(path)
    # print(spk_emb.shape)


