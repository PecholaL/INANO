""" INANO
    * Content encoder (EncC) and decoder (Dec) are taken from VC model (AdaIN-VC/MAIN-VC)
    * Generator (G) is based on GLOW
    ** EncC and Dec are trained via the VC model's reconstruction process
    ** G is trained on the speaker encoder dataset output by the speaker encoder of the VC model
    ! Note: All the modules mentioned above are trained and INANO only emsembles the pre-trained modules.
    ! i.e. INANO does NOT need additional training.
    """

import yaml
from MAIN_VC.main_vc import ContentEncoder, Decoder
from Generator import getFlow


class INANO:
    def __init__(self, vc_config_path, g_config_path):
        with open(vc_config_path, "r") as f:
            self.vc_config = yaml.load(f, Loader=yaml.FullLoader)
        with open(g_config_path, "r") as f:
            self.g_config = yaml.load(f, Loader=yaml.FullLoader)

        self.EncC = ContentEncoder(self.vc_config["ContentEncoder"])
        self.Dec = Decoder(self.vc_config["Decoder"])
        self.G = getGlow(self.g_config["Generator"])

    def forward(self, orig_mel):
        cnt_emb = self.EncC(orig_mel)
        pseudo_spk_emb = self.G.sample(self.g_config["Generator"]["spk_emb_dim"])
        ano_mel = self.Dec(cnt_emb, pseudo_spk_emb)
        return ano_mel
