import os

from ._dataclass import SV_Trial, SV_TrainItem, SV_EnrollmentItem

class VoxCeleb1:
    NUM_TRAIN_ITEM = 148642
    NUM_TRAIN_SPK = 1211
    NUM_TRIALS = 37611

    def __init__(self, path_train, path_test, path_trials):
        self.train_set = []
        self.test_set_O = []
        self.trials_O = []
        self.class_weight = []

        # train_set
        labels = {}
        num_utt = [0 for _ in range(self.NUM_TRAIN_SPK)]
        num_sample = 0
        for root, _, files in os.walk(path_train):
            for file in files:
                if '.wav' in file:
                    # combine path
                    f = os.path.join(root, file)
                    
                    # parse speaker
                    spk = f.split('/')[-3]
                    
                    # labeling
                    try: labels[spk]
                    except: 
                        labels[spk] = len(labels.keys())

                    # init item
                    item = SV_TrainItem(path=f, speaker=spk, label=labels[spk])
                    self.train_set.append(item)
                    num_sample += 1
                    num_utt[labels[spk]] += 1

        for n in num_utt:
            self.class_weight.append(num_sample / n)
                    
        # test_set
        for root, _, files in os.walk(os.path.join(path_test, 'test')):
            for file in files:
                if '.wav' in file:
                    f = os.path.join(root, file)
                    item = SV_EnrollmentItem(path=f, key='/'.join(f.split('/')[-3:]))
                    self.test_set_O.append(item)

        self.trials_O = self.parse_trials(os.path.join(path_trials, 'trials.txt'))

        # error check
        assert len(self.train_set) == self.NUM_TRAIN_ITEM
        assert len(self.trials_O) == self.NUM_TRIALS
        assert len(labels) == self.NUM_TRAIN_SPK

    def parse_trials(self, path):
        trials = []
        for line in open(path).readlines():
            strI = line.split(' ')
            item = SV_Trial(
                key1=strI[1].replace('\n', ''), 
                key2=strI[2].replace('\n', ''), 
                label=int(strI[0])
            )
            trials.append(item)
        return trials