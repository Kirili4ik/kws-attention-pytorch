import sys
import torch
import torchaudio
import matplotlib.pyplot as plt

from models import CRNN, AttnMech, ApplyAttn



# win_len=400, hop_len=200
def get_mel_len(audio):
    return int((audio.size(0) - 400)/200) + 3


if __name__ == '__main__':
    N_MELS     = 40

    IN_SIZE = 40
    HIDDEN_SIZE = 64
    KERNEL_SIZE = (20, 5)
    STRIDE = (8, 2)
    GRU_NUM_LAYERS = 2
    NUM_DIRS = 2
    NUM_CLASSES = 2

    kernel_x = KERNEL_SIZE[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_audio, sr = torchaudio.load(sys.argv[1])
    test_audio = test_audio.squeeze()

    # Create models
    CRNN_model = CRNN(IN_SIZE, HIDDEN_SIZE, KERNEL_SIZE, STRIDE, GRU_NUM_LAYERS)
    attn_layer = AttnMech(HIDDEN_SIZE * NUM_DIRS)
    apply_attn = ApplyAttn(HIDDEN_SIZE * 2, NUM_CLASSES)
    # Load models
    checkpoint = torch.load('crnn_final', map_location=device)
    CRNN_model.load_state_dict(checkpoint['model_state_dict'])
    checkpoint = torch.load('attn_final', map_location=device)
    attn_layer.load_state_dict(checkpoint['model_state_dict'])
    checkpoint = torch.load('apply_attn_final', map_location=device)
    apply_attn.load_state_dict(checkpoint['model_state_dict'])

    # Create melspec
    melspec_test = torchaudio.transforms.MelSpectrogram(
        sample_rate=48000,
        n_mels=N_MELS
    ).to(device)


    # TEST
    all_probs = []

    CRNN_model.eval()
    attn_layer.eval()
    apply_attn.eval()
    is_kw = False
    with torch.no_grad():
        start = 41
        finish = get_mel_len(test_audio)

        test_audio_mel = torch.log(melspec_test(test_audio) + 1e-9).unsqueeze(0).to(device)
        hidden = torch.zeros(GRU_NUM_LAYERS*2, 1, HIDDEN_SIZE).to(device)    # (num_layers*num_dirs, BS, HS)

        # apply full model, but save crnn_outputs, **e** and hidden
        e = []
        outputs, hidden = CRNN_model(test_audio_mel[:, :, 0 : start], hidden)
        for el in outputs:
            e_t = attn_layer(el)
            e.append(e_t)
        new_e = torch.cat(e, dim=1)
        probs = apply_attn(new_e, outputs)
        # for plotting
        all_probs.append(torch.exp(probs[1]))

        end = (finish - start + 1)
        start -= kernel_x
        for i in range(kernel_x, end, kernel_x):
            # delete first element
            e = e[1:]
            outputs = outputs[1:]

            # get next frame with size 5
            batch_now = test_audio_mel[:, :, start + i : start + i + kernel_x]

            # apply to batch with (seq_len=1; batch_size=1)
            output, hidden = CRNN_model(batch_now, hidden)   # hidden is also new!
            # output: (1, BS, hidden*num_dir)

            # add new crnn_output to previous
            outputs = torch.cat([outputs, output])

            # recount attention
            e_t = attn_layer(output.squeeze(0))
            e.append(e_t)
            new_e = torch.cat(e, dim=1)

            # apply_attention
            probs = apply_attn(new_e, outputs)
            # if > 0.5 then there is a word probably
            prob_now = torch.exp(probs[1])
            if prob_now > 0.5:
                is_kw = True

            # save for logging
            all_probs.append(prob_now)

    if is_kw:
        print('There IS probably a keyword in your sentence! Look at graph ' + sys.argv[1] + '.pdf')
    else:
        print('There is probably NO keyword in your sentence. But it\'s better to look at graph ' + sys.argv[1] + '.pdf')
    f = plt.figure()
    plt.xlabel('time mel frame')
    plt.ylabel('probability')
    plt.hlines(y=0.3, xmin=0, xmax=get_mel_len(test_audio)/5 - 8, linestyles='--', colors='r')
    plt.plot(all_probs)
    f.savefig(sys.argv[1] + '.pdf')
