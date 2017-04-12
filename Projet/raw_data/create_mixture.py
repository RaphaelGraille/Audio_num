import numpy as np
import librosa
import soundfile as sf
import os
import sys
import glob
from random import randint, uniform
from math import exp, log

speech = "LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac"
noise = "env_sounds/car1.wav"

ENV_NOISE_PATH = "./env_sounds/"
SPEECH_PATH ="LibriSpeech/dev-clean"
MIXTURE_PATH = "../data/mixtures_train/"
SPEECH_TRAIN_PATH = "../data/speech_train/"

MIXTURE_TEST_PATH = "../data/mixtures_test/"
SPEECH_TEST_PATH = "../data/speech_test/"

NB_SPEAKER_TRAIN = 35
NB_SPEAKER_TEST = 5

MAX_SNR = 5

SAMPLING_RATE = 8000
MIXTURE_TIME = 5 # Time of the created mixture in s 
MIXTURE_SIZE = SAMPLING_RATE*MIXTURE_TIME # size of the created mixture in nb of sample

def normalize_signal_affine(signal, targetMin = -1, targetMax = 1):
	minS = signal.min()
	maxS = signal.max()
	return (signal-minS)*(targetMax - targetMin)/(maxS - minS) + targetMin

def normalize_signal_max(signal):
	return signal/(np.absolute(signal)).max()

def load_sound(file_name, sr=22050):
	data, samplerate = sf.read(file_name, dtype='float32')
	data = data.T
	data_22k = librosa.resample(data, samplerate, sr)
	return data_22k, sr

#def select_random_noise():
#	noise_envs = os.listdir(ENV_NOISE_PATH)
#	rand_index = randint(0,len(noise_envs)-1)
#	return ENV_NOISE_PATH+noise_envs[rand_index]

def select_random_noise(noises_array):
	idx = randint(0,len(noises_array)-1)
	return noises_array[idx]

def select_random_audio_part(noise_signal, speech_signal, part_size=MIXTURE_SIZE):
	start_index = randint(0,len(noise_signal)-len(speech_signal))
	return noise_signal[start_index:(start_index+len(speech_signal))]

def signal_parts(signal, part_size=MIXTURE_SIZE):
	nb_parts = len(signal)/part_size
	parts = []
	for i in range(nb_parts):
		part = signal[i*part_size:(i+1)*part_size]
		parts.append(part)
	yield parts

def load_env_sounds():
	noises_array = []
	noises = os.listdir(ENV_NOISE_PATH)
	for noise in noises:
		if noise[0] != ".":
			noise_signal, _ = load_sound(ENV_NOISE_PATH+noise,sr=SAMPLING_RATE)
			noises_array.append(noise_signal)
	return noises_array


def mix_signals(signal, noise, SNR):
	noise = noise/(exp(log(10)*SNR/10.0))
	return(signal+noise)/2.0

def get_all_speech(speech_path=SPEECH_PATH):
	all_speech_files = []
	books = os.listdir(SPEECH_PATH)
	#for book in books[0:NB_SPEAKER_TRAIN]:
	for book in books[NB_SPEAKER_TRAIN:NB_SPEAKER_TRAIN+NB_SPEAKER_TEST]:
		full_path = os.path.join(SPEECH_PATH,book)
		chapters = os.listdir(full_path)
		for chapter in chapters:
			full_path = os.path.join(SPEECH_PATH,book,chapter)
			audio_files = [full_path + "/" + f for f in os.listdir(full_path) if (f[0]!="." and f[(len(f)-5):len(f)]==".flac")]
			all_speech_files = all_speech_files + audio_files
	return all_speech_files

def create_mixtures():
	print("loading noises ...")
	noises_array = load_env_sounds()
	print("done")

	all_speech_files = get_all_speech()
	for speech in all_speech_files:
		speech_signal, _ = load_sound(speech,sr=SAMPLING_RATE)
		speech_signal = normalize_signal_max(speech_signal)
		i = 0
		#for speech_signal_part in signal_parts(speech_signal):
		noise_signal = select_random_noise(noises_array)
		#noise_signal, _ = load_sound(noise,sr=SAMPLING_RATE)
		noise_part = select_random_audio_part(noise_signal,speech_signal)
		noise_part = normalize_signal_max(noise_part)
		SNR = uniform(0,MAX_SNR)
		mixed_signal = mix_signals(speech_signal, noise_part, SNR)
		output_name = MIXTURE_TEST_PATH+os.path.basename(speech)[0:(len(os.path.basename(speech))-5)] +".wav" # +"_"+format(i,'04d')+".wav"
		output_name_speech = SPEECH_TEST_PATH+os.path.basename(speech)[0:(len(os.path.basename(speech))-5)] +".wav"
		librosa.output.write_wav(output_name,mixed_signal,SAMPLING_RATE)
		librosa.output.write_wav(output_name_speech,speech_signal,SAMPLING_RATE)


def test_function():
	print("loading files...")
	ys, sr1 = load_sound(speech, sr=8000)
	yn, sr2 = load_sound(noise, sr=8000)

	#yn = normalize_signal(yn)
	#ys = normalize_signal(ys)

	yn = normalize_signal_max(yn)
	ys = normalize_signal_max(ys)

	all_speech_files = get_all_speech()

	print("loading noises")
	env_sounds = load_env_sounds()
	print(len(env_sounds))
	print(env_sounds)

	if sr1 == sr2:
		print("Mixing files ...")	
		#ym = mix_signals(ys[0:24000],yn[0:24000],1)
		#librosa.output.write_wav('test_mixture.wav', ym, sr1)

#test_function()

create_mixtures()
