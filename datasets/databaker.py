from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import os
from datasets import mel


def build_from_path(hparams, input_dir, mel_dir, linear_dir, wav_dir, n_jobs=12, tqdm=lambda x: x):
	executor = ProcessPoolExecutor(max_workers=n_jobs)
	futures = []
	index = 1
	with open(os.path.join(input_dir, 'metadata.csv'), encoding='utf-8') as f:
		for line in f:
			parts = line.strip().split('|')
			basename = parts[0]
			wav_path = os.path.join(input_dir, 'wavs', '{}.wav'.format(basename))
			text = parts[2]
			futures.append(executor.submit(partial(_process_utterance, mel_dir, linear_dir, wav_dir, basename, wav_path, text, hparams)))
			index += 1

	return [future.result() for future in tqdm(futures) if future.result() is not None]


def _process_utterance(mel_dir, linear_dir, wav_dir, index, wav_path, text, hparams):
	if os.path.exists(wav_path):
		mel_spectrogram, linear_spectrogram, out = mel.wav2mel(wav_path)
		time_steps = len(out)
		mel_frames = mel_spectrogram.shape[0]
	else:
		print('file {} present in csv metadata is not present in wav folder. skipping!'.format(wav_path))
		return None

	# Write the spectrogram and audio to disk
	audio_filename = 'audio-{}.npy'.format(index)
	mel_filename = 'mel-{}.npy'.format(index)
	linear_filename = 'linear-{}.npy'.format(index)
	np.save(os.path.join(wav_dir, audio_filename), out.astype(np.float32), allow_pickle=False)
	np.save(os.path.join(mel_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)
	np.save(os.path.join(linear_dir, linear_filename), linear_spectrogram.T, allow_pickle=False)

	# Return a tuple describing this training example
	return (audio_filename, mel_filename, linear_filename, time_steps, mel_frames, text)
