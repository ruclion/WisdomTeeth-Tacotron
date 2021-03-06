import os
import torch
import numpy as np
from torch.utils.data import DataLoader


from hparams import hparams
from models import Tacotron2
from dataset import TextMelLoader, TextMelCollate
from preprocess_dataset.gl import write_wav, mel2wav




device = 'cuda' 
torch.manual_seed(hparams.seed)
torch.cuda.manual_seed(hparams.seed)



def inference():
    # data
    testset = TextMelLoader(hparams.mel_training_files, hparams)
    test_loader = DataLoader(testset, num_workers=0, shuffle=False, batch_size=1, collate_fn=TextMelCollate)


    # model
    model = Tacotron2(hparams)
    checkpoint_dict = torch.load(hparams.checkpoint_path)
    model.load_state_dict(checkpoint_dict['state_dict'])
    model.cuda().eval()


    # iteration
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # flow [1]:  data to 'cuda'
            text_padded, _input_lengths, _mel_padded, _gate_padded, _output_lengths = batch
            text_padded = text_padded.to(device).long()
            

            # flow [2]:  predict
            _mels_pre, mels_pre_postnet, _gates_pre, _alignments_pre = model(text_padded=text_padded, text_lengths=None, mel_padded=None)
            

            # flow [3]: to wav, and in mel2wav, will write
            wav_path = os.path.join(hparams.mel_test_files, '{}.wav'.format(i))
            _wav_pre = mel2wav(mels_pre_postnet[0], wav_name_path=wav_path)


            # flow [4]: save 错误形式 mel-npy
            mels = mels_pre_postnet[0].cpu().numpy()
            mel_path = os.path.join(hparams.mel_test_files, '{}.npy'.format(i))
            # 存在硬盘的为错误形式, (80, T_out)
            np.save(mel_path, mels.T, allow_pickle=False)




if __name__ == '__main__':
    inference()