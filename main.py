import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchaudio
import wandb

from my_utils import set_seed, get_sampler, count_parameters, transform_tr
from my_utils import preprocess_data as my_collate_fn
from dataset  import TrainDataset
from models import CRNN, AttnMech, ApplyAttn, FullModel
from train_val import train_epoch, validation


if __name__ == '__main__':
    BATCH_SIZE = 256
    NUM_EPOCHS = 35
    N_MELS     = 40

    IN_SIZE = 40
    HIDDEN_SIZE = 64
    KERNEL_SIZE = (20, 5)
    STRIDE = (8, 2)
    GRU_NUM_LAYERS = 2
    NUM_DIRS = 2
    NUM_CLASSES = 2

    set_seed(21)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Working on', device)

    ### Dataset
    my_dataset = TrainDataset(csv_path='labels_sheila.csv', transform=transform_tr)
    print('All train+val samples:', len(my_dataset))
    train_len = 57500
    val_len = 64721 - train_len
    train_set, val_set = torch.utils.data.random_split(my_dataset, [train_len, val_len])
    # Samplers for oversampling
    train_sampler = get_sampler(train_set.dataset.csv['label'][train_set.indices].values)
    val_sampler   = get_sampler(val_set.dataset.csv['label'][val_set.indices].values)
    # Loaders
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=False, collate_fn=my_collate_fn,
                              sampler=train_sampler, drop_last=False,
                              num_workers=1, pin_memory=True)

    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=my_collate_fn,
                            sampler=val_sampler, drop_last=False,
                            num_workers=1, pin_memory=True)

    ### Create melspecs
    # With augmentations
    melspec_train = nn.Sequential(
        torchaudio.transforms.MelSpectrogram(sample_rate=16000,  n_mels=N_MELS),
        torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
        torchaudio.transforms.TimeMasking(time_mask_param=35),
    ).to(device)
    # W/o augmentations
    melspec_val = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_mels=N_MELS
    ).to(device)

    ### Create model
    CRNN_model = CRNN(IN_SIZE, HIDDEN_SIZE, KERNEL_SIZE, STRIDE, GRU_NUM_LAYERS)
    attn_layer = AttnMech(HIDDEN_SIZE * NUM_DIRS)
    apply_attn = ApplyAttn(HIDDEN_SIZE * 2, NUM_CLASSES)

    checkpoint = torch.load('crnn_final', map_location=device)
    CRNN_model.load_state_dict(checkpoint['model_state_dict'])
    checkpoint = torch.load('attn_final', map_location=device)
    attn_layer.load_state_dict(checkpoint['model_state_dict'])
    checkpoint = torch.load('apply_attn_final', map_location=device)
    apply_attn.load_state_dict(checkpoint['model_state_dict'])

    full_model = FullModel(CRNN_model, attn_layer, apply_attn)
    print(full_model.to(device))
    print(count_parameters(full_model))
    wandb.init()
    wandb.watch(full_model)

    ### Create optimizer
    opt = torch.optim.Adam(full_model.parameters(), weight_decay=1e-5)

    ### Train_val loop
    for n in range(NUM_EPOCHS):
        train_epoch(full_model, opt, train_loader, melspec_train,
              GRU_NUM_LAYERS, HIDDEN_SIZE, device=device)

        validation(full_model, val_loader, melspec_val,
            GRU_NUM_LAYERS, HIDDEN_SIZE, device=device)

        print('END OF EPOCH', n)

    ### Save model
#    torch.save({
#        'model_state_dict': CRNN_model.state_dict(),
#    }, 'crnn_final')

#    torch.save({
#        'model_state_dict': attn_layer.state_dict(),
#    }, 'attn_final')

#    torch.save({
#        'model_state_dict': apply_attn.state_dict(),
#    }, 'apply_attn_final')

    validation(full_model, val_loader, melspec_val,
            GRU_NUM_LAYERS, HIDDEN_SIZE, device=device, find_trsh=True)
