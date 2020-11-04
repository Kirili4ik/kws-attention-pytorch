import torch
from torch.utils.data import WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from torch import distributions

import torchaudio
import numpy as np
import random



def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def add_rand_noise(audio):
    background_noises = [
        'speech_commands/_background_noise_/white_noise.wav',
       'speech_commands/_background_noise_/dude_miaowing.wav',
       'speech_commands/_background_noise_/doing_the_dishes.wav',
       'speech_commands/_background_noise_/exercise_bike.wav',
       'speech_commands/_background_noise_/pink_noise.wav',
       'speech_commands/_background_noise_/running_tap.wav'
    ]

    noise_num = torch.randint(low=0, high=len(background_noises), size=(1,)).item()
    noise = torchaudio.load(background_noises[noise_num])[0].squeeze()

    noise_level = torch.Tensor([1])   # [0, 40]
    noise_energy = torch.norm(noise)
    audio_energy = torch.norm(audio)
    alpha = (audio_energy / noise_energy) * torch.pow(10, -noise_level / 20)

    start = torch.randint(low=0, high=int(noise.size(0) - audio.size(0) - 1), size=(1,)).item()
    noise_sample = noise[start : start + audio.shape[0]]

    audio_new = audio + alpha * noise_sample
    audio_new.clamp_(-1, 1)
    return audio_new


def transform_tr(wav):
    aug_num = torch.randint(low=0, high=4, size=(1,)).item()
    augs = [
        lambda x: x,
        lambda x: (x + distributions.Normal(0, 0.01).sample(x.size())).clamp_(-1, 1),
        lambda x: torchaudio.transforms.Vol(.25)(x),
        lambda x: add_rand_noise(x)
    ]
    return augs[aug_num](wav)


def get_sampler(target):
    class_sample_count = np.array(
        [len(np.where(target == t)[0]) for t in np.unique(target)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in target])

    samples_weight = torch.from_numpy(samples_weight)
    samples_weigth = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler


# collate_fn actually
def preprocess_data(data):
    wavs = []
    labels = []

    for el in data:
        wavs.append(el['utt'])
        labels.append(el['label'])

    wavs = pad_sequence(wavs, batch_first=True)
    labels = torch.Tensor(labels).type(torch.long)
    return wavs, labels


def count_FA_FR(preds, labels):
    FA = torch.sum(preds[labels == 0])
    FR = torch.sum(labels[preds == 0])
    return FA.item()/torch.numel(preds), FR.item()/torch.numel(preds)


def get_au_fa_fr(probs, labels, device, find_trsh=False):
    max_F1 = -1

    sorted_probs, indices = torch.sort(probs)
    sorted_probs = torch.cat((torch.Tensor([0]), sorted_probs))
    sorted_probs = torch.cat((sorted_probs, torch.Tensor([1])))
    labels = torch.cat(labels, dim=0)

    FAs, FRs = [], []
    for prob in sorted_probs:
        ones = (probs >= prob) * 1
        FA, FR = count_FA_FR(ones, labels)

        if find_trsh:
            F1 = 2 * (FA * FR) / (FA + FR)
            print(F1, prob)
            if F1 > max_F1:
                max_F1 = F1
                best_trsh = prob

        FAs.append(FA)
        FRs.append(FR)
    # plt.plot(FAs, FRs)
    # plt.show()
    if find_trsh:
        return best_trsh, -np.trapz(FRs, x=FAs)
    return -np.trapz(FRs, x=FAs)


