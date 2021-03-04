from scipy.io import wavfile
from scipy import signal
import numpy as np
import librosa


def write_wav(write_path, wav_arr, sr):
    wav_arr *= 32767 / max(0.01, np.max(np.abs(wav_arr)))
    wavfile.write(write_path, sr, wav_arr.astype(np.int16))
    return


def _db_denormalize(normalized_db, min_db):
    # 只写了[-4, 4]版本
    return (np.clip(normalized_db, -4., 4.) + 4.) / 8. * (-min_db) + min_db


def _db2mag(mag_db, ref_db):
    return np.power(10.0, (mag_db + ref_db)/20)


def _mag_mel2mag_spec(mag_mel, sr, n_fft, num_mels, fmin, fmax):
    mag_mel_t = mag_mel.T

    global _mel_basis, _inv_mel_basis
    _mel_basis = librosa.filters.mel(sr, n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)   # [n_mels, 1+n_fft/2]
    _inv_mel_basis = np.linalg.pinv(_mel_basis) 
    mag_spec_t = np.dot(_inv_mel_basis, mag_mel_t)
    mag_spec_t = np.maximum(1e-10, mag_spec_t)
    mag_spec = mag_spec_t.T

    return mag_spec


def _stft(wav_arr, n_fft, hop_len, win_len):
    return librosa.core.stft(wav_arr, n_fft=n_fft, hop_length=hop_len,
                             win_length=win_len)


def _istft(stft_matrix, hop_len, win_len):
    return librosa.core.istft(stft_matrix, hop_length=hop_len,
                              win_length=win_len)


def _griffin_lim(magnitude_spec, gl_iterations, n_fft, hop_len, win_len):
    # # 在这里进行gl的power，输入的是正常的magnitude_spec
    # magnitude_spec = magnitude_spec ** gl_power
    mag = magnitude_spec.T  # transpose to [n_freqs, time]
    angles = np.exp(2j * np.pi * np.random.rand(*mag.shape))
    complex_mag = np.abs(mag).astype(np.complex)
    stft_0 = complex_mag * angles
    y = _istft(stft_0, hop_len = hop_len, win_len = win_len)
    for _i in range(gl_iterations):
        angles = np.exp(1j * np.angle(_stft(y, n_fft=n_fft, hop_len=hop_len, win_len=win_len)))
        y = _istft(complex_mag * angles, hop_len = hop_len, win_len = win_len)
    return y


def _deemphasis(wav_arr, pre_param):
    return signal.lfilter([1], [1, -pre_param], wav_arr)



def mel2wav(normalized_db_mel, sr=16000, preemphasis=0.85, 
                n_fft=2048, hop_len=200,
                win_len=800, num_mels=80, 
                fmin=0.,
                fmax=8000.,
                ref_db=20, min_db=-115,
                griffin_lim_power=1.5,
                griffin_lim_iterations=60, wav_name_path='a.wav'):
    assert normalized_db_mel.shape[-1] == 80
    db_mel = _db_denormalize(normalized_db_mel, min_db=min_db)
    mag_mel = _db2mag(db_mel, ref_db=ref_db)
    mag_spec = _mag_mel2mag_spec(mag_mel, sr=sr, n_fft=n_fft, num_mels=num_mels, fmin=fmin, fmax=fmax) #矩阵求逆猜出来的spec
    magnitude_spec = mag_spec ** 1.0 # (time, n_fft/2+1)
    griffinlim_powered_magnitude_spec = magnitude_spec ** griffin_lim_power # (time, n_fft/2+1)
    emph_wav_arr = _griffin_lim(griffinlim_powered_magnitude_spec, gl_iterations=griffin_lim_iterations,
                                n_fft=n_fft, hop_len=hop_len, win_len=win_len)
    wav_arr = _deemphasis(emph_wav_arr, pre_param=preemphasis)

    write_wav(wav_name_path, wav_arr, 16000)
    return wav_arr

if __name__ == '__main__':
    mel_path = 'training_data/mels/mel-000003.npy'
    a = np.load(mel_path).T
    print(a.shape)
    b = mel2wav(a)
    print(b)
    print(b.shape)

