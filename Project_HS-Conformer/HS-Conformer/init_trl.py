import os

trl = '/data/ASVspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
new = '/data/ASVspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train_augment.trn.txt'
flac = '/data/ASVspoof2019/LA/ASVspoof2019_LA_train/'

f = open(new, 'w')
for line in open(trl).readlines():
    strI = line.replace('\n', '').split(' ')
    
    ext = []
    for d in os.listdir(flac):
        if '.' in d:
            ext.append(d)
    
    for e in ext:
        sample = f'{strI[0]} {strI[1]} {strI[2]} {strI[3]} {strI[4]}\n'
        f.write(strI[0])
        f.write(' ')
        


    print(os.listdir(flac))
    
    
    
    exit()
     
    
    
    

    f = os.path.join(path_train, self.PATH_TRAIN_FLAC, f'{strI[1]}.flac')
    attack_type = strI[3]
    label = 0 if strI[4] == 'bonafide' else 1
    if label == 0:
        train_num_neg += 1
    else:
        train_num_pos += 1
        
    item = DF_Item(f, label, attack_type, is_fake=(label == 1))
    self.train_set.append(item)