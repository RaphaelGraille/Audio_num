import numpy as np
import librosa
import soundfile as sf
import os
import sys

SPEECH_TRAIN_FOLDER = "./speech_train/"
MIXTURE_TRAIN_FOLDER = "./mixtures_train/"

SPEECH_TEST_FOLDER = "./speech_test/"
MIXTURE_TEST_FOLDER = "./mixtures_test/"

NB_CONC_FRAMES=20 #number of frames concatenated to be fed in the NN

NB_MAX_FRAME = 140000/NB_CONC_FRAMES # number max ofconcatenated frames in each file

N_FFT = 256

SAMPLING_RATE = 8000

def load_sound(file_name, sr=22050):
	data, samplerate = sf.read(file_name, dtype='float32')
	data = data.T
	data_22k = librosa.resample(data, samplerate, sr)
	return data_22k, sr

def concatenate_frames(frames, nb_conc=NB_CONC_FRAMES):
	new_size = frames.shape[1]/nb_conc
	if (frames.shape[1] % nb_conc != 0):
		new_size = new_size + 1
	concatenated_frames = np.zeros((nb_conc*frames.shape[0],new_size))
	#for i in range(0,new_size):
	#	for j in range(0,nb_conc):
	#		concatenated_frames[i,j:j+nb_conc] = frames[:,j]
	#print(concatenated_frames)
	for i in range(0,frames.shape[1]):
		#print(str(i))
		#rint(frames[:,i])
		concatenated_frames[(i%nb_conc)*frames.shape[0]:((i%nb_conc)+1)*frames.shape[0],i/nb_conc] = frames[:,i]
	return concatenated_frames

def deconcatenate_frames(frames, nb_conc=NB_CONC_FRAMES):
    deconcatenated_frames = np.zeros((frames.shape[0]*nb_conc,frames.shape[1]/nb_conc))
    for i in range(0,frames.shape[0]):
        for j in range(0,nb_conc):
        	deconcatenated_frames[i*nb_conc+j,:] = frames[i,j*(frames.shape[1]/nb_conc):(j+1)*(frames.shape[1]/nb_conc)]
    return deconcatenated_frames

def create_data_and_labels():

	train_data = []
	train_labels = []
	index = 1
	for mixture in os.listdir(MIXTURE_TRAIN_FOLDER):
		if len(train_data) > NB_MAX_FRAME:
			print("Saving date to file ...")
			train_data_array = np.array(train_data)
			train_labels_array = np.array(train_labels)

			data = np.zeros((2,train_data_array.shape[0],train_data_array.shape[1]))
			data[0] = train_data_array
			data[1] = train_labels_array
			np.save("./train_data_and_labels_"+str(index)+".npy",data)
			#np.save("./train_data_"+str(index)+".npy",np.array(train_data))
			#np.save("./train_labels_"+str(index)+".npy",np.array(train_labels))
			#print("loading data...")
			#data_load = np.load("./train_data_and_labels_"+str(index)+".npy")
			#print("done loading")
			#t0 = data_load[0]
			#t1 = data_load[1]

			#if np.array_equal(t0,train_data_array) and np.array_equal(t1,train_labels_array):
			#	print("YOUPIIII")

			train_data = []
			train_labels = []
			index = index + 1
			print("Done !")
		if mixture[0] != ".":
			print(mixture)
			mixture_signal,_ = load_sound(MIXTURE_TRAIN_FOLDER+mixture,sr=SAMPLING_RATE)
			speech_signal,_ = load_sound(SPEECH_TRAIN_FOLDER+mixture,sr=SAMPLING_RATE)
			mixture_fft = librosa.core.stft(mixture_signal, n_fft=N_FFT)
			speech_fft = librosa.core.stft(speech_signal, n_fft=N_FFT)

			mixture_magn = np.abs(mixture_fft)
			speech_magn = np.abs(speech_fft)

			print(mixture_magn.shape)
			print(speech_magn.shape)

			mixture_magn = concatenate_frames(mixture_magn)
			speech_magn = concatenate_frames(speech_magn)

			print(mixture_magn.shape)
			print(speech_magn.shape)

			for i in range(0,mixture_magn.shape[1]):
				mixture_frame_i = mixture_magn[:,i]
				speech_frame_i = speech_magn[:,i]

				train_data.append(mixture_frame_i)
				train_labels.append(speech_frame_i)
	if train_data != []:
		#np.save("./train_data_"+str(index)+".npy",np.array(train_data))
		#np.save("./train_labels_"+str(index)+".npy",np.array(train_labels))
		print("Saving date to file ...")
		train_data_array = np.array(train_data)
		train_labels_array = np.array(train_labels)

		data = np.zeros((2,train_data_array.shape[0],train_data_array.shape[1]))
		data[0] = train_data_array
		data[1] = train_labels_array
		np.save("./train_data_and_labels_"+str(index)+".npy",data)

def save_data_and_labels():
	print("creating data and labels ...")
	train_data, train_labels = create_data_and_labels()
	print("saving data and labels ...")
	np.save("./train_data.npy",train_data)
	np.save("./train_labels.npy",train_labels)

#create_data_and_labels()
test = np.array(range(40)).reshape((4,10))
print(test)
res = concatenate_frames(test,nb_conc=3)
print(np.transpose(res))
res = deconcatenate_frames(np.transpose(res),nb_conc=3)
print(res)

