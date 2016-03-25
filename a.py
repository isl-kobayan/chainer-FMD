import numpy as np
import random
import os
import six
import matplotlib
# Force matplotlib to not use any Xwindows backend.
#matplotlib.use("Agg")
import matplotlib.pyplot as plt

out_dir="./g1"
os.mkdir(out_dir)
data=np.loadtxt("W.txt", delimiter="\t")
data=data.transpose()
x=np.arange(1, 11)
lim=0.25
#labels=('fabric','foliage','glass','leather','metal','paper','plastic','stone','water','wood')
labels=('0','1','2','3','4','5','6','7','8','9')
for i in six.moves.range(1024):
    fig = plt.figure(figsize=(4.2,3.5))
    fig.subplots_adjust(top=0.95, left=0.15, right=0.95, bottom = 0.12)
    plt.clf()
    #lim=np.abs(data[i]).max()
    plt.bar(x,data[i], align="center")
    plt.xlim(0.5,10.5)
    plt.ylim(-lim,lim)
    #plt.xticks([])
    plt.xticks(x, labels[:10], fontsize=30, rotation=0)
    plt.yticks([-lim,0,lim], fontsize=20)

    #plt.barh(x,data[i], align="center")
    #plt.ylim(0.5,10.5)
    #plt.xlim(-lim,lim)
    #plt.yticks([])
    #plt.yticks(x, labels[:10], rotation=0)
    #plt.xticks([-lim,0,lim])
    #plt.legend(scatterpoints=1)
    plt.savefig(out_dir + "/" + str(i).zfill(4)+".eps", format="eps")
    plt.close()
    print(i)
