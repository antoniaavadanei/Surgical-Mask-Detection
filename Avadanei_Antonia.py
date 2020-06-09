from sklearn import svm
import librosa.display
import numpy as np
from sklearn import preprocessing
import csv
train_rows = []
train_files_names = []
train_labels = []
test_files_names = []

#Function used to normalize data with min_max
def normalize(train_features, test_features):

    min_max_scaler = preprocessing.MinMaxScaler()

    min_max_scaler.fit(train_features)
    #normalizing train data
    scaled_train_feats = min_max_scaler.transform(train_features)
    # normalizing test data
    scaled_test_feats = min_max_scaler.transform(test_features)

    return scaled_train_feats, scaled_test_feats

#function used to extract features using librosa.load and Mel-frequency cepstral coefficients
def extract_features(path, audio_files):
    feats = []  # list of all train features

    for audio_file in audio_files:
        file = path + audio_file
        time_series, sampling_rate = librosa.load(file, res_type='kaiser_fast')
        mel_frequency = librosa.feature.mfcc(time_series, n_mfcc=200, sr=sampling_rate)

        scaled_mfccs = np.mean(mel_frequency.T, axis=0)
        feats.append(scaled_mfccs)

    feats = np.array(feats)
    return feats


#uploading train data and labels
with open('../train.txt', newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        train_rows.append(row)
for train_file_name, train_label in train_rows:
    train_files_names.append(train_file_name)
    train_labels.append(train_label)


#uploading test data
with open('../test.txt', newline='') as f:
    for file in f:
        test_files_names.append(file[:-1])


path_train= "/proiect/train/train\\"
path_test = "/proiect/test/test\\"

#extracting train features and test festures
tr_feats=extract_features(path_train,train_files_names)
tst_feats=extract_features(path_test,test_files_names)

#converting training label list to numpy array
tr_labels = np.array(train_labels)

#normalizing the data with min_max
scaled_train, scaled_test= normalize(tr_feats, tst_feats)

#creating the model
svm_model = svm.SVC(C=1, kernel='linear')


#fitting data
svm_model.fit(scaled_train, tr_labels)


#predicting labels
predictions = svm_model.predict(scaled_test)


#writing predictions to file
file_to_write = open("../kaggle11.txt", "a")
file_to_write.write("name,label\n")
for i in range(len(test_files_names)):
    file_to_write.write(str(test_files_names[i]) + "," + predictions[i])
    file_to_write.write("\n")

print("Finish")
