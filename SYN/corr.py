import numpy as np

clip1=np.load('/home/zhenyao/SYN_CMP2/clips3.npy')
clip2=np.load('/home/zhenyao/SYN_CMP2/clips4.npy')
corr = np.arange(64).reshape(8, 8)

for i in range(8):
    for j in range(8):
        corr[i][j]=0
        for p in range(64):
            corr[i][j] += (clip1[i][p]-clip2[j][p])*(clip1[i][p]-clip2[j][p])*100000000
print corr
