import os
import torch
import random
import numpy as np


_pad = '_'
_eos = '~'
_tone = '123456'
_letters = 'abcdefghijklmnopqrstuvwxyz'
_space = ' '


# Export all symbols:
symbols = [_pad] + [_eos] + list(_tone) + list(_letters) + [_space]


_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}



def text_to_sequence(text):
    res =  [_symbol_to_id[s] for s in text]
    res.append(_symbol_to_id['~'])
    return res


def sequence_to_text(sequence):
  result = ''
  for symbol_id in sequence:
    if symbol_id in _id_to_symbol:
      s = _id_to_symbol[symbol_id]
      result += s
  return result



class TextMelDataset(torch.utils.data.Dataset):
    """
        1) loads filepath,text pairs
        2) change text to symbol id sequence
        3) loads mel-spectrograms from mel files
    """
    def __init__(self, fname, hparams):
        self.f_list = self.files_to_list(fname)
        random.seed(hparams.seed)
        random.shuffle(self.f_list)


    def files_to_list(self, file_path):
        f_list = []
        with open(file_path, encoding = 'utf-8') as f:
            for line in f.readlines():
                parts = line.strip().split('|') 
                # mel_file_path
                path  = parts[1]
                # text
                text  = parts[5]
                # print(text)
                f_list.append([text, path])
        return f_list


    def get_mel_text_pair(self, text, file_path):
        text = self.get_text(text)
        mel = self.get_mel(file_path)
        return (text, mel)


    def get_mel(self, file_path):
        # 便于习惯, 均修正为 seq_first, 即 (B, T_out, num_mels)
        # stored melspec: np.ndarray [shape=(T_out, num_mels)]
        mel_abs_path = os.path.join('/ceph/home/hujk17/WisdomTeeth-Tacotron/preprocess_dataset/training_data/mels', file_path)
        melspec = torch.from_numpy(np.load(mel_abs_path))
        # assert melspec.shape[-1] == 80
        return melspec


    def get_text(self, text):
        text_norm = torch.tensor(np.asarray(text_to_sequence(text)))
        return text_norm


    def __getitem__(self, index):
        a = self.get_mel_text_pair(self.f_list[index][0], self.f_list[index][1])
        print('inner:', a)
        return a


    def __len__(self):
        return len(self.f_list)





class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per step
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)

        return text_padded, input_lengths, mel_padded, gate_padded, output_lengths


if __name__ == '__main__':
    print(len(symbols))