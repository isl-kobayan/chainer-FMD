from PIL import Image
import numpy as np
import random
import os
import six
import matplotlib
# Force matplotlib to not use any Xwindows backend.
#matplotlib.use("Agg")
import matplotlib.pyplot as plt

def read_image(path, insize, mean_image=None, center=False, flip=False):
    # Data loading routine
    image = np.asarray(Image.open(path).convert('RGB')).transpose(2, 0, 1)[::-1]

    cropwidth = image.shape[1] - insize
    if center:
        top = left = cropwidth / 2
    else:
        top = random.randint(0, cropwidth - 1)
        left = random.randint(0, cropwidth - 1)
    bottom = insize + top
    right = insize + left

    image = image[:, top:bottom, left:right].astype(np.float32)

    if mean_image is not None:
        image -= mean_image[:, top:bottom, left:right]

    image /= 255
    if flip and random.randint(0, 1) == 0:
        return image[:, :, ::-1]
    else:
        return image

def read_crop_image(path, insize, mean_image=None, flip=False):
    # get image data as np.float32
    image = np.asarray(Image.open(path).convert('RGB')).transpose(2, 0, 1)[::-1]

    cropwidth = image.shape[1] - insize
    #if center:
    #    top = left = cropwidth / 2
    #    else:
    top = random.randint(0, cropwidth - 1)
    left = random.randint(0, cropwidth - 1)
    bottom = insize + top
    right = insize + left

    image = image[:, top:bottom, left:right].astype(np.float32)
    if mean_image is not None:
        image -= mean_image[:, top:bottom, left:right]
    image /= 255
    if flip and random.randint(0, 1) == 0:
        return image[:, :, ::-1]
    else:
        return image

def load_num2label(path):
    tuples = []
    for line in open(path):
        pair = line.strip().split()
        tuples.append(pair[0])
    return tuples

def load_image_list(path, root='.'):
    tuples = []
    for line in open(path):
        pair = line.strip().split()
        tuples.append((os.path.join(root, pair[0]), np.int32(pair[1])))
    return tuples

def save_confusion_matrix(path, matrix, labels):
    labelsize = len(labels)
    matrix = matrix.reshape((labelsize, labelsize))
    content = 'true\\pred'
    for labelname in labels:
        content = content + '\t' + labelname
    content = content + '\n'
    for true_idx in six.moves.range(labelsize):
        content = content + labels[true_idx]
        for pred_idx in six.moves.range(labelsize):
            content = content + '\t' + str(int(matrix[true_idx, pred_idx]))
        content = content + '\n'
    with open(path, 'w') as f:
        f.write(content)

def append_acts(path, imagepath, acts):
    content = imagepath + '\t'
    for i in six.moves.range(len(acts)):
       content = content + str(acts[i]) + '\t'
    content = content + '\n'
    with open(path, 'a') as f:
        f.write(content)

def get_act_table(path, val_list, acts, rank=10):
    maxargs=None
    for r in six.moves.range(rank):
        maxarg = np.argmax(acts, axis=0)
        if maxargs is None:
            maxargs=maxarg.reshape((1, -1))
        else:
            maxargs=np.r_[maxargs, maxarg.reshape((1, -1))]
        for i in six.moves.range(len(maxarg)):
            acts[maxarg[i],i]=0
    maxargs = maxargs.transpose()
    content=''
    for r in six.moves.range(maxargs.shape[0]):
        for i in six.moves.range(maxargs.shape[1]):
            content = content + val_list[maxargs[r,i]][0] + '\t'
        content = content + '\n'
    with open(path, 'a') as f:
        f.write(content)



# save confusion matrix as .png image
def save_confmat_fig0(conf_arr, savename, labels):
    norm_conf = []
    for i in conf_arr:
        #print(i)
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet,
                interpolation='nearest')

    width = len(conf_arr)
    height = len(conf_arr[0])

    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(conf_arr[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

    cb = fig.colorbar(res)
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    plt.xticks(range(width), labels[:width])
    plt.yticks(range(height), labels[:height])
    plt.savefig(savename, format='png')


def save_confmat_fig(conf_arr, savename, labels,
                     xlabel=None, ylabel=None, saveFormat="png",
                     title=None, clim=(None,None), mode="vote", cmap=plt.cm.Blues):
    if mode=="rate":
        conf_rate = []
        for i in conf_arr:
            tmp_arr = []
            total = float(sum(i))
            for j in i:
                if total == 0:
                    tmp_arr.append(float(j))
                else:
                    tmp_arr.append(float(j)/total)
            conf_rate.append(tmp_arr)
        conf_arr = conf_rate
    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            if a == 0:
                tmp_arr.append(float(j))
            else:
                tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)
    fig = plt.figure()
    plt.clf()
    plt.subplots_adjust(top=0.85) # use a lower number to make more vertical space
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    if mode == "rate":
        res = plt.imshow(np.array(norm_conf)*100, cmap=cmap,
                        interpolation='nearest')
        plt.clim(0,100)
        threshold = 0.5
    else:
        res = plt.imshow(np.array(norm_conf), cmap=cmap,
                        interpolation='nearest')
        if clim!=(None,None):
            plt.clim(*clim)
        threshold = np.mean([np.max(norm_conf),np.min(norm_conf)])
    width = len(conf_arr)
    height = len(conf_arr[0])

    for x in xrange(width):
        for y in xrange(height):
            if norm_conf[x][y]>=threshold:
                textcolor = '1.0'
            else:
                textcolor = '0.0'
            if mode == "rate":
                ax.annotate("{0:d}".format(int(conf_arr[x][y]*100)), xy=(y, x),
                            horizontalalignment='center',
                            verticalalignment='center', color=textcolor, fontsize=15)
            else:
                ax.annotate("{0}".format(conf_arr[x][y]), xy=(y, x),
                            horizontalalignment='center',
                            verticalalignment='center', color=textcolor, fontsize=15)

    cb = fig.colorbar(res)
    cb.ax.tick_params(labelsize=15)
    if title != None:
        plt.text(0.5, 1.08, title,
                 horizontalalignment='center',
                 fontsize=20,
                 transform = ax.transAxes)
    ax.xaxis.tick_top()
    plt.xticks(range(width), labels[:width], rotation=45, fontsize=15)
    plt.yticks(range(height), labels[:height], fontsize=15)
    if xlabel != None:
        plt.xlabel(xlabel)
    if ylabel != None:
        plt.ylabel(ylabel)
        ax.xaxis.set_label_position("top")
    plt.savefig(savename, format=saveFormat)
    plt.close(fig)

def save_pca_scatter_fig(data, savename, labels,
                     xlabel=None, ylabel=None, saveFormat="png",
                     title=None, cmap=plt.cm.Blues):
    import sklearn.decomposition
    import matplotlib.cm as cm
    markers=['x', '+']
    data=data.reshape((len(data), -1))
    pca = sklearn.decomposition.PCA(n_components = 2)
    pca.fit(data)
    result = pca.transform(data)
    np.savetxt(savename + ".result.csv", result, delimiter=",")
    print(result.shape)
    size1 = len(data)/len(labels)
    result = result.reshape((len(labels), size1, 2))
    fig = plt.figure()
    plt.clf()
    for c in six.moves.range(len(labels)):
        #print(result[c, :, 0].shape)
        #print(result[c, :, 0])
        plt.scatter(result[c, :, 0], result[c, :, 1], marker=markers[c%len(markers)], color=cm.jet(float(c) / len(labels)), label=labels[c])
    plt.legend(scatterpoints=1)
    plt.savefig(savename, format=saveFormat)
    plt.close()
