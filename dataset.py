import json
import random
import codecs
import numpy as np
import torch
import torch.utils.data

from utils import load_filepaths_and_text


def get_mel_text_pair(melpath_and_text):
        # separate filename and text
        melpath = melpath_and_text[0]
        text = melpath_and_text[1]
        mel = torch.from_numpy(np.load(melpath))
        return (text, mel)


class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, melpaths_and_text, hparams):
        self.melpaths_and_text = load_filepaths_and_text(melpaths_and_text)
        random.seed(hparams.seed)
        random.shuffle(self.melpaths_and_text)


    def __getitem__(self, index):
        return get_mel_text_pair(self.melpaths_and_text[index])


    def __len__(self):
        return len(self.melpaths_and_text)




# class TextMelCollate():
#     """ Zero-pads model inputs and targets based on number of frames per setep
#     """
#     def __init__(self, n_frames_per_step):
#         self.n_frames_per_step = n_frames_per_step

#     def __call__(self, batch):
#         """Collate's training batch from normalized text and mel-spectrogram
#         PARAMS
#         ------
#         batch: [text_normalized, mel_normalized]
#         """
#         # Right zero-pad all one-hot text sequences to max input length
#         input_lengths, ids_sorted_decreasing = torch.sort(
#             torch.LongTensor([len(x[0]) for x in batch]),
#             dim=0, descending=True)
#         max_input_len = input_lengths[0]

#         inputs_padded = torch.LongTensor(len(batch), max_input_len)
#         inputs_padded.zero_()
#         for i in range(len(ids_sorted_decreasing)):
#             input_id = batch[ids_sorted_decreasing[i]][0]
#             inputs_padded[i, :input_id.shape[0]] = input_id

#         phonemes_padded = torch.LongTensor(len(batch), max_input_len)
#         phonemes_padded.zero_()
#         for i in range(len(ids_sorted_decreasing)):
#             phoneme_id = batch[ids_sorted_decreasing[i]][1]
#             phonemes_padded[i, :phoneme_id.shape[0]] = phoneme_id

#         # Right zero-pad mel-spec
#         num_mels = batch[0][2].size(0)
#         max_target_len = max([x[2].size(1) for x in batch])
#         if max_target_len % self.n_frames_per_step != 0:
#             max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
#             assert max_target_len % self.n_frames_per_step == 0

#         # include mel padded and gate padded
#         mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
#         mel_padded.zero_()
#         gate_padded = torch.FloatTensor(len(batch), max_target_len)
#         gate_padded.zero_()
#         output_lengths = torch.LongTensor(len(batch))
#         for i in range(len(ids_sorted_decreasing)):
#             mel = batch[ids_sorted_decreasing[i]][2]
#             mel_padded[i, :, :mel.size(1)] = mel
#             gate_padded[i, mel.size(1)-1:] = 1
#             output_lengths[i] = mel.size(1)

#         return input_lengths, inputs_padded, phonemes_padded, mel_padded, gate_padded, output_lengths