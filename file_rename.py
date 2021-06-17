import glob
import os

path = '検証データ/*.jpg'
i = 1

flist = glob.glob(path)

for file in flist:
    os.rename(file, '検証データ/'+str(i)+'.jpg')
    i=i+1