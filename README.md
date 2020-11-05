# Key word spotting task (KWS attention)

Config:
Model described in Shan et al., 2018 (Attention-based End-to-End Models for Small-Footprint Keyword Spotting; https://arxiv.org/abs/1803.10916)

Model is ~144k parameters, but can be changed with parameters. 
Config in main:
    
    BATCH_SIZE = 256        (size of batch for learning)
    NUM_EPOCHS = 35         (number of epochs to train model)
    N_MELS     = 40         (number of mels for melspectrogram)

    IN_SIZE = 40            (size of input)
    HIDDEN_SIZE = 64        (size of hidden representation in 
    KERNEL_SIZE = (20, 5)   (size of kernel for convolution layer in CRNN)
    STRIDE = (8, 2)         (size of stride for convolution layer in CRNN)
    GRU_NUM_LAYERS = 2      (number of GRU layers in CRNN)
    NUM_DIRS = 2            (number of directions in GRU (2 if bidirectional))
    NUM_CLASSES = 2         (number of classes (2 for "no word" or "sheila is in audio")

Data for training can be downloaded here:
http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz

For learning run "python main.py".

Code supports inferense as described in paper. Install requirements and use "python inference.py YOUR_AUDIO". It works even on cpu fast and easy. 
Script generates "YOUR_AUDIO.pdf" with graph of probabilities of word "Sheila" being said on audio.  
