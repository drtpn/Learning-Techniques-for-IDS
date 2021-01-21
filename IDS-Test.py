import numpy as np
import pandas as pd

df=pd.read_csv('./KDDTrain+.csv')

y=df['label']
"""
def unique(y):
    # intilize a null list
    unique_list = []

    # traverse for all elements
    for x in y:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
            # print list
    for x in unique_list:
        print (x),


print("the unique values from 1st list is")
unique(y)
"""
n=(df.label == 'normal').sum()
print('Normal: ',(df.label == 'normal').sum())
print('buffer_overflow: ',(df.label == 'buffer_overflow').sum())
print('loadmodule: ',(df.label == 'loadmodule').sum())
print('perl: ',(df.label == 'perl').sum())
print('neptune: ',(df.label == 'neptune').sum())
print('smurf: ',(df.label == 'smurf').sum())
print('guess_passwd: ',(df.label == 'guess_passwd').sum())
print('pod: ',(df.label == 'pod').sum())
print('teardrop: ',(df.label == 'teardrop').sum())
print('portsweep: ',(df.label == 'portsweep').sum())
print('ipsweep: ',(df.label == 'ipsweep').sum())
print('land: ',(df.label == 'land').sum())
print('ftp_write: ',(df.label == 'ftp_write').sum())
print('Back: ',(df.label == 'back').sum())
print('imap: ',(df.label == 'imap').sum())
print('satan: ',(df.label == 'satan').sum())
print('phf: ',(df.label == 'phf').sum())
print('nmap: ',(df.label == 'nmap').sum())
print('multihop: ',(df.label == 'multihop').sum())
print('warezmaster: ',(df.label == 'warezmaster').sum())
print('warezclient: ',(df.label == 'warezclient').sum())
print('Spy: ',(df.label == 'spy').sum())
print('Rootkit: ',(df.label == 'rootkit').sum())

dos= (df.label == 'back').sum() + (df.label == 'land').sum() + (df.label == 'smurf').sum() + (df.label == 'pod').sum() + (df.label == 'neptune').sum() + (df.label == 'teardrop').sum()
print('DOS: ',dos)

probe= (df.label == 'ipsweep').sum() + (df.label == 'portsweep').sum() + (df.label == 'nmap').sum() + (df.label == 'satan').sum()
print('Probe: ',probe)

u2r= (df.label == 'loadmodule').sum() + (df.label == 'rootkit').sum() + (df.label == 'perl').sum() + (df.label == 'buffer_overflow').sum()
print('U2R: ',u2r)

r2l= (df.label == 'guess_passwd').sum() + (df.label == 'multihop').sum() + (df.label == 'ftp_write').sum() + (df.label == 'spy').sum() + (df.label == 'phf').sum() + (df.label == 'imap').sum() + (df.label == 'warezmaster').sum() + (df.label == 'warezclient').sum()
print('R2L: ',r2l)

print('Total', dos+probe+u2r+r2l+n)