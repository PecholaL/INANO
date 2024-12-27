# INANO
Instance Normalization-based Speaker Anonymization  

## Speech Representation Disentanglement
The pre-trained model used for speech representation disentanglement (SRD) is [MAIN-VC](https://github.com/PecholaL/MAIN-VC), which disentangles speaker representation and content representation from speech. The disentangled speaker representations are used for training the Glow-based pseudo speaker representation generator, which generates the pseudo speaker representation for anonymization. INANO completes the generation of pseudo speaker representation and the synthesis of anonimized speech with the pseudo speaker representation.  

## Pseudo Speaker Representation
This model leverage a FLOW for generating speaker representation, learning the distribution of the target from the speaker representation disentangled via the pre-trained MAIN-VC. Then during the inference stage, the Peseudo Spk Generator takes in Gaussian noise and produces speaker representation for anonymization.

## Anonymization
The Decoder which generates the Mel-spectrogram of the anonymized speech is the same to the decoder in MAIN-VC. A pre-trained vocoder is also leveraged for the generation of waveforms.

## Training
### prepare
Firstly, these files should be obtained: *attr.pkl, config.yaml, mainVcModel.ckpt, vocoder.pt*. Among them, *attr.pkl* is obtained from the data preprocess of MAIN-VC, *mainVcModel.ckpt* is obtained after MAIN-VC's training, *vocoder.pt* is available at [here](https://drive.google.com/file/d/1r0exien35ddN303dtYdCriHwDxVSFY_7/view?usp=sharing), *config.yaml* is available at [here](https://github.com/PecholaL/MAIN-VC/blob/main/config.yaml). Please modify the paths to these files according to your preparation in the main funtion of *mainvc_inf.py*. 

### Obtain Speaker Embeddings
To obtain real speaker embeddings for generator's training, the pre-trained MAIN-VC is utilized to disentangle the speaker embeddings. Set **speech_corpus_root** and **spk_emb_dataset_path** in the *mainvc_inf.py*. Then directly excecute:
```python mainvc_inf.py```
to obtain speaker embedding dataset.