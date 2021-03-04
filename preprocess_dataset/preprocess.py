import argparse
import os
from multiprocessing import cpu_count

# from tqdm import tqdm
import numpy as np
import mel

from concurrent.futures import ProcessPoolExecutor
from functools import partial



# in
original_txt_path = '/ceph/dataset/DataBaker.BZNSYP/ProsodyLabeling/000001-010000.txt'
wav_dir_path = '/ceph/dataset/DataBaker.DB6/CN/wav'

# out
mel_dir = os.path.join('training_data', 'mels')
wav_dir = os.path.join('training_data', 'audio')
lin_dir = os.path.join('training_data', 'linear')
os.makedirs(mel_dir, exist_ok=True)
os.makedirs(wav_dir, exist_ok=True)
os.makedirs(lin_dir, exist_ok=True)
out_txt_path = ['training_data/train.txt', 'training_data/val.txt', 'training_data/test.txt']




def _process_utterance(mel_dir, linear_dir, wav_dir, index, wav_path, text):
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



def main():
	executor = ProcessPoolExecutor(max_workers=cpu_count())
	futures = []
	with open(original_txt_path, encoding='utf-8') as f:
		# 000001	卡尔普#2陪外孙#1玩滑梯#4。
		# ka2 er2 pu3 pei2 wai4 sun1 wan2 hua2 ti1
		f_list = f.readlines()
		i = 0
		while i < len(f_list):
			basename = f_list[i].strip()[:6]
			wav_path = os.path.join(wav_dir_path, '{}.wav'.format(basename))
			text = f_list[i + 1].strip()

			futures.append(executor.submit(partial(_process_utterance, mel_dir, lin_dir, wav_dir, basename, wav_path, text)))
			i += 2
			# break

		metadata = [future.result() for future in futures if future.result() is not None]
	
	start = [0, 8000, 9000]
	endd = [7999, 8999, 9999]
	for k in range(3):
		with open(out_txt_path[k], 'w', encoding='utf-8') as f:
			for i in range(start[k], endd[k]):
				f.write('|'.join([str(x) for x in metadata[i]]) + '\n')
	
	print('Write {} utterances'.format(len(metadata)))
	print('Max input length (text chars): {}'.format(max(len(m[5]) for m in metadata)))
	print('Max mel frames length: {}'.format(max(int(m[4]) for m in metadata)))


if __name__ == '__main__':
	main()
