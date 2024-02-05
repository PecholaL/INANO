# SRD_SpkAno
Speech Representation Disentanglement-based speaker anonymization model.

## SRD
The model used for speech representation disentanglement is [MAIN-VC](https://github.com/PecholaL/MAIN-VC), which disentangles speaker representation and content representation from speech. This SpkAno model completes the generation of pseudo speaker representation and the synthesis of anonimized speech with the pseudo speaker representation.

## Pseudo Spk Representation
This model leverage a GAN for generating speaker representation, learning the distribution of the target from the speaker representation disentangled via the pre-trained MAIN-VC. Then during the inference stage, the Peseudo Spk Generator takes in Gaussian noise and produces speaker representation for anonymization.

## Anonymization
The Decoder which generates the Mel-spectrogram of the anonymized speech is the same to the decoder in MAIN-VC. A pre-trained vocoder is also leveraged for the generation of waveforms.