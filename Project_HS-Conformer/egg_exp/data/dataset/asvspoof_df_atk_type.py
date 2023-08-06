import os

from ._dataclass import DF_Item

class ASVspoof2021_DF_Atk_Type:
    NUM_TEST_ITEM   = 611829

    PATH_TRAIN_TRL  = 'LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
    # HS fix here
    PATH_TRAIN_TRL_DA = 'LA/ASVspoof2019_LA_cm_protocols/metadata_with_DA.txt'  # HS
    PATH_TRAIN_FLAC = 'LA/ASVspoof2019_LA_train'
    PATH_TEST_TRL   = 'keys/DF/CM/trial_metadata.txt'
    PATH_TEST_FLAC  = 'ASVspoof2021_DF_eval/flac'

    # HS fix here
    def __init__(self, path_train, path_test, DA=True, print_info=False, atk_type='codec', only_spoof=False):   # HS
        self.test_set = []
        self.atk_type = {}
        self.num_atk_type = 0

        # # train_set
        # train_num_pos = 0
        # train_num_neg = 0
        # # HS fix here
        # trl = os.path.join(path_train, self.PATH_TRAIN_TRL_DA if DA else self.PATH_TRAIN_TRL)   # HS
        # for line in open(trl).readlines():
        #     strI = line.replace('\n', '').split(' ')

        #     f = os.path.join(path_train, self.PATH_TRAIN_FLAC, f'{strI[1]}.flac')
        #     attack_type = strI[3]
        #     label = 0 if strI[4] == 'bonafide' else 1   # Real: 0, Fake: 1
        #     if label == 0:
        #         train_num_neg += 1
        #     else:
        #         train_num_pos += 1
                
        #     item = DF_Item(f, label, attack_type, is_fake=(label == 1))
        #     self.train_set.append(item)

        # self.class_weight.append((train_num_neg + train_num_pos) / train_num_neg)
        # self.class_weight.append((train_num_neg + train_num_pos) / train_num_pos)
        
        # test_set
        test_num_pos = 0
        test_num_neg = 0
        trl = os.path.join(path_test, self.PATH_TEST_TRL)
        for line in open(trl).readlines():
            strI = line.replace('\n', '').split(' ')
            f = os.path.join(path_test, self.PATH_TEST_FLAC, f'{strI[1]}.flac')
            if atk_type == 'codec':
                attack_type = strI[2]
            elif atk_type == 'vocoder':
                attack_type = strI[8]
            label = 0 if strI[4] == '-' else 1
            if label == 0:
                test_num_neg += 1
            else:
                test_num_pos += 1
                
            item = DF_Item(f, label, attack_type, is_fake=label == 1)
            self.test_set.append(item)
            
            try: 
                self.atk_type[attack_type].append(item)
            except:
                self.atk_type[attack_type] = [item]
                self.num_atk_type += 1
        
        if atk_type == 'vocoder' and not only_spoof:
            bonafide = self.atk_type['bonafide']
            # print(len(bonafide))
            self.atk_type.pop('bonafide')
            atk_types = list(self.atk_type.keys())
            for key in atk_types:
                self.atk_type[key].extend(bonafide)
                print(key, len(self.atk_type[key]))

        # error check
        assert len(self.test_set) == self.NUM_TEST_ITEM, f'[DATASET ERROR] - TEST_SAMPLE: {len(self.test_set)}, EXPECTED: {self.NUM_TEST_ITEM}'

        # print info
        if print_info:
            info = (
                  f'====================\n'
                + f'    ASVspoof2021    \n'
                + f'====================\n'
                + f'TEST  (ASVspoof2021 DF):  bona - {test_num_neg}, spoof - {test_num_pos}\n'
                + f'====================\n'
            )
            print(info)