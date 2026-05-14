# %%
# %pip install torch
# %pip install torchaudio
# %pip install tqdm
# %pip install librosa
# %pip install numpy
# %pip install miditoolkit


# %%
import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from tqdm import tqdm
import librosa
import numpy as np
import miditoolkit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, average_precision_score, accuracy_score
import random

def write_submission_predictions(predictions, outpath, normalize_audio_paths=False):
    # The autograder reads these files with eval(...), so write Python literals.
    serializable = predictions
    if normalize_audio_paths:
        serializable = {
            (k[2:] if isinstance(k, str) and k.startswith('./') else k): v
            for k, v in predictions.items()
        }
    with open(outpath, "w") as z:
        z.write(repr(serializable) + '\n')
    return serializable

# %% [markdown]
# ## Metrics

# %%
def accuracy1(groundtruth, predictions):
    correct = 0
    for k in groundtruth:
        if not (k in predictions):
            print("Missing " + str(k) + " from predictions")
            return 0
        if predictions[k] == groundtruth[k]:
            correct += 1
    return correct / len(groundtruth)

# %%
def accuracy2(groundtruth, predictions):
    correct = 0
    for k in groundtruth:
        if not (k in predictions):
            print("Missing " + str(k) + " from predictions")
            return 0
        if predictions[k] == groundtruth[k]:
            correct += 1
    return correct / len(groundtruth)

# %%
TAGS = ['rock', 'oldies', 'jazz', 'pop', 'dance',  'blues',  'punk', 'chill', 'electronic', 'country']

# %%
def accuracy3(groundtruth, predictions):
    preds, targets = [], []
    for k in groundtruth:
        if not (k in predictions):
            print("Missing " + str(k) + " from predictions")
            return 0
        prediction = [predictions[k][tag] for tag in TAGS]
        target = [1 if tag in groundtruth[k] else 0 for tag in TAGS]
        preds.append(prediction)
        targets.append(target)

    mAP = average_precision_score(targets, preds, average='macro')
    return mAP

# %% [markdown]
# ## Task 1: Composer classification

# %%
# dataroot1 = "student_files_updated/task1_composer_classification"
dataroot1 = r"C:\Users\Seojin Park\Desktop\Coding\CSE 153 Assignment1\student_files_updated\student_files\task1_composer_classification"

# %%
class model1():
    def __init__(self):
        pass

    def features(self, path):
        midi_obj = miditoolkit.midi.parser.MidiFile(dataroot1 + '/' + path)
        notes = midi_obj.instruments[0].notes
        num_notes = len(notes)
        average_pitch = sum([note.pitch for note in notes]) / num_notes
        average_duration = sum([note.end - note.start for note in notes]) / num_notes
        
        # pitch_classes = [0] * 12
        # for note in notes:
        #     pitch_classes[note.pitch % 12] += 1
            
        # #normalize first
        # pitch_classes = [x / num_notes for x in pitch_classes]
        # # more freatures
        # std_pitch = np.sqrt(
        #     sum([(note.pitch - average_pitch) ** 2 for note in notes]) / num_notes
        # )
        # std_duration = np.sqrt(
        #     sum([((note.end - note.start) - average_duration) ** 2 for note in notes]) / num_notes
        # )
        
        #for convinience 
        pitches = np.array([n.pitch for n in notes])
        durations = np.array([n.end - n.start for n in notes])
        starts = np.array([n.start for n in notes])
        
        avg_dur = np.mean(durations)
        std_dur = np.std(durations)
        num_notes = len(notes)
        pitch_range = np.max(pitches) - np.min(pitches)
        
        pitch_classes = np.bincount(pitches % 12, minlength=12)
        pitch_classes = pitch_classes / num_notes
        
        #melodic interval feats
        if len(pitches) > 1:
            intervals = np.diff(pitches)
            abs_intervals = np.abs(intervals)

            interval_mean = np.mean(intervals)
            interval_std = np.std(intervals)
            abs_interval_mean = np.mean(abs_intervals)
            abs_interval_std = np.std(abs_intervals)
        else:
            interval_mean = interval_std = abs_interval_mean = abs_interval_std = 0
        
        
        #histogram feature
        interval_hist, z = np.histogram(
            np.clip(intervals if len(pitches) > 1 else [0], -12, 12),
            bins=12,
            range=(-12, 12),
            density=True
        )
        
        
        
        # timing
        onsets = [0]
        if len(starts) > 1:
            onsets = np.diff(np.sort(starts))
        onset_mean = 0
        onset_std = 0
        if len(onsets):
            onset_mean = np.mean(onsets)
            onset_std = np.std(onsets)
        
        
        # temportal seg, idk if this even makes a diff T_T
        t_min, t_max = np.min(starts), np.max(starts)
        thirds = np.linspace(t_min, t_max + 1e-6, 4)
        
        segment_feats = []
        for i in range(3):
            mask = (starts >= thirds[i]) & (starts < thirds[i+1])
            if np.sum(mask) > 0:
                seg_pitches = pitches[mask]
                segment_feats += [
                    np.mean(seg_pitches),
                    np.std(seg_pitches),
                    len(seg_pitches)
                ]
            else:
                segment_feats += [0, 0, 0]
                
                
                
                
                
                
                
                
                
        features = np.concatenate([
            pitch_classes,
            interval_hist,
            [
                onset_mean,
                onset_std,
                avg_dur,
                std_dur,
                num_notes,
                pitch_range,
                interval_mean,
                average_pitch,
                average_duration,
                interval_std,
                abs_interval_mean,
                abs_interval_std
            ],
            segment_feats
        ])

        return features.tolist()

    def predict(self, path, outpath=None):
        d = eval(open(path, 'r').read())
        predictions = {}
        for k in d:
            x = self.features(k)
            pred = self.model.predict([x])
            predictions[k] = int(pred[0])
        if outpath:
            predictions = write_submission_predictions(predictions, outpath)
        return predictions

    # Train your model. Note that this function will not be called from the autograder:
    # instead you should upload your saved model using save()
    def train(self, path):
        with open(path, 'r') as f:
            train_json = eval(f.read())
        X_train = [self.features(k) for k in train_json]
        y_train = [int(train_json[k]) for k in train_json]
        model = RandomForestClassifier(n_estimators = 200, max_depth= None, random_state= 10)
        model.fit(X_train, y_train)
        self.model = model

# %% [markdown]
# ## Task 2: Sequence prediction

# %%
# dataroot2 = "student_files/task2_next_sequence_prediction/"
dataroot2 = r"C:\Users\Seojin Park\Desktop\Coding\CSE 153 Assignment1\student_files_updated\student_files\task1_composer_classification"

# %%
class model2():
    def __init__(self):
        pass

    def features(self, path):
        midi_obj = miditoolkit.midi.parser.MidiFile(dataroot2 + '/' + path)
        notes = midi_obj.instruments[0].notes
        
        min_tick = min([note.start for note in notes])
        max_tick = max([note.start for note in notes])
        notes_with_min_tick = [note.pitch for note in notes if note.start == min_tick]
        notes_with_max_tick = [note.pitch for note in notes if note.start == max_tick]
        avg_pitch_min_tick = sum(notes_with_min_tick) / len(notes_with_min_tick)
        avg_pitch_max_tick = sum(notes_with_max_tick) / len(notes_with_max_tick)
        return avg_pitch_min_tick, avg_pitch_max_tick

    def train(self, path):
        # This baseline doesn't use any model (it just measures pitch difference)
        # You can use this approach but *probably* you'll want to implement a model
        pass

    def predict(self, path, outpath=None):
        d = eval(open(path, 'r').read())
        predictions = {}
        for k in tqdm(d):
            path1, path2 = k # Keys are pairs of paths
            x1_min, x1_max = self.features(path1)
            x2_min, x2_max = self.features(path2)
            # Given two segments, it compares:
                # - how close the end of segment 1 is to the beginning of segment 2 in terms of pitch
                # - how close the end of segment 2 is to the beginning of segment 1 in terms of pitch
            if abs(x1_min - x2_max) > abs(x2_min - x1_max):
                predictions[k] = True
            else:
                predictions[k] = False
        if outpath:
            predictions = write_submission_predictions(predictions, outpath)
        return predictions

# %% [markdown]
# ## Task 3: Audio classification

# %%
SAMPLE_RATE = 22050
N_MELS = 64
N_CLASSES = 10
AUDIO_DURATION = 10
BATCH_SIZE = 32

# %%
dataroot3 = "student_files/task3_audio_classification/"

# %%
def extract_waveform(path):
    waveform, sr = librosa.load(dataroot3 + '/' + path, sr=SAMPLE_RATE)
    waveform = torch.FloatTensor(np.array([waveform]))  # 텐서로 먼저 변환
    if sr != SAMPLE_RATE:
        resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = resample(waveform)
    # Pad so that everything is the right length
    target_len = SAMPLE_RATE * AUDIO_DURATION
    if waveform.shape[1] < target_len:
        pad_len = target_len - waveform.shape[1]
        waveform = F.pad(waveform, (0, pad_len))
    else:
        waveform = waveform[:, :target_len]
    return waveform

# %%
class AudioDataset(Dataset):
    def __init__(self, meta, preload = True):
        self.meta = meta
        ks = list(meta.keys())
        self.idToPath = dict(zip(range(len(ks)), ks))
        self.pathToFeat = {}

        self.mel = MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=N_MELS)
        self.db = AmplitudeToDB()

        self.preload = preload # Determines whether the features should be preloaded (uses more memory)
                               # or read from disk / computed each time (slow if your system is i/o-bound)
        if self.preload:
            for path in tqdm(ks, desc="Preloading audio"):
                waveform = extract_waveform(path)
                mel_spec = self.db(self.mel(waveform)).squeeze(0)
                self.pathToFeat[path] = mel_spec

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        # Faster version, preloads the features
        path = self.idToPath[idx]
        tags = self.meta[path]
        bin_label = torch.tensor([1 if tag in tags else 0 for tag in TAGS], dtype=torch.float32)

        if self.preload:
            mel_spec = self.pathToFeat[path]
        else:
            waveform = extract_waveform(path)
            mel_spec = self.db(self.mel(waveform)).squeeze(0)

        return mel_spec.unsqueeze(0), bin_label, path

# %%
class Loaders():
    def __init__(self, train_path, test_path, split_ratio=0.9, seed = 0):
        torch.manual_seed(seed)
        random.seed(seed)

        meta_train = eval(open(train_path, 'r').read())
        l_test = eval(open(test_path, 'r').read())
        meta_test = dict([(x,[]) for x in l_test])

        print("Loading train set...")
        all_train = AudioDataset(meta_train)
        print("Loading test set...")
        test_set = AudioDataset(meta_test)

        # Split all_train into train + valid
        total_len = len(all_train)
        train_len = int(total_len * split_ratio)
        valid_len = total_len - train_len
        train_set, valid_set = random_split(all_train, [train_len, valid_len])

        self.loaderTrain = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        self.loaderValid = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        self.loaderTest = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# %%
class CNNClassifier(nn.Module):
    def __init__(self, n_classes=N_CLASSES):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        # torchaudio MelSpectrogram defaults: n_fft=400, hop_length=200
        # time frames = SAMPLE_RATE * AUDIO_DURATION // 200 + 1
        n_time = SAMPLE_RATE * AUDIO_DURATION // 200 + 1
        self.fc1 = nn.Linear(32 * (N_MELS // 4) * (n_time // 4), 256)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (B, 16, mel/2, time/2)
        x = self.pool(F.relu(self.conv2(x)))  # (B, 32, mel/4, time/4)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return torch.sigmoid(self.fc2(x))  # multilabel → sigmoid

# %%
class Pipeline():
    def __init__(self, model, learning_rate, seed = 0):
        # These two lines will (mostly) make things deterministic.
        # You're welcome to modify them to try to get a better solution.
        torch.manual_seed(seed)
        random.seed(seed)

        self.device = torch.device("cpu") # Can change this if you have a GPU, but the autograder will use CPU
        self.model = model.to(self.device) #model.cuda() # Also uncomment these lines for GPU
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()

    def evaluate(self, loader, threshold=0.5, outpath=None):
        self.model.eval()
        preds, targets, paths = [], [], []
        with torch.no_grad():
            for x, y, ps in tqdm(loader, desc="Evaluating"):
                x = x.to(self.device) #x.cuda()
                y = y.to(self.device) #y.cuda()
                outputs = self.model(x)
                preds.append(outputs.cpu())
                targets.append(y.cpu())
                paths += list(ps)

        preds = torch.cat(preds)
        targets = torch.cat(targets)

        predictions = {}
        for i in range(preds.shape[0]):
            predictions[paths[i]] = {TAGS[j]: float(preds[i][j]) for j in range(len(TAGS))}

        mAP = None
        if outpath: # Save predictions
            predictions = write_submission_predictions(predictions, outpath, normalize_audio_paths=True)
        else: # Only compute accuracy if we're *not* saving predictions, since we can't compute test accuracy
            mAP = average_precision_score(targets, preds, average='macro')
        return predictions, mAP

    def train(self, train_loader, val_loader, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for x, y, path in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                x = x.to(self.device) #x.cuda()
                y = y.to(self.device) #y.cuda()
                self.optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            val_predictions, mAP = self.evaluate(val_loader)
            print(f"[Epoch {epoch+1}] Loss: {running_loss/len(train_loader):.4f} | Val mAP: {mAP:.4f}")

# %% [markdown]
# ## Run everything...

# %%
def run1():
    model = model1()
    model.train(dataroot1 + "/train.json")
    train_preds = model.predict(dataroot1 + "/train.json")
    test_preds = model.predict(dataroot1 + "/test.json", "predictions1.json")

    train_labels = eval(open(dataroot1 + "/train.json").read())
    acc1 = accuracy1(train_labels, train_preds)
    print("Task 1 training accuracy = " + str(acc1))

# %%
def run2():
    model = model2()
    model.train(dataroot2 + "/train.json")
    train_preds = model.predict(dataroot2 + "/train.json")
    test_preds = model.predict(dataroot2 + "/test.json", "predictions2.json")

    train_labels = eval(open(dataroot2 + "/train.json").read())
    acc2 = accuracy2(train_labels, train_preds)
    print("Task 2 training accuracy = " + str(acc2))

# %%
def run3():
    loaders = Loaders(dataroot3 + "/train.json", dataroot3 + "/test.json")
    model = CNNClassifier()
    pipeline = Pipeline(model, 1e-4)

    pipeline.train(loaders.loaderTrain, loaders.loaderValid, 5)
    train_preds, train_mAP = pipeline.evaluate(loaders.loaderTrain, 0.5)
    valid_preds, valid_mAP = pipeline.evaluate(loaders.loaderValid, 0.5)
    test_preds, _ = pipeline.evaluate(loaders.loaderTest, 0.5, "predictions3.json")

    all_train = eval(open(dataroot3 + "/train.json").read())
    for k in valid_preds:
        # We split our training set into train+valid
        # so need to remove validation instances from the training set for evaluation
        all_train.pop(k)
    acc3 = accuracy3(all_train, train_preds)
    print("Task 3 training mAP = " + str(acc3))

# %%
run1()

# %%
run2()

# %%
run3()

# %%



