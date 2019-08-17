
import math
import os
import tensorflow as tf
import time
import numpy as np
import matplotlib as plt
from matplotlib import *
from sklearn import manifold, datasets
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib import figure
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import tensorflow as tf
from PIL import Image

import numpy as np
from PIL import ImageDraw
# We may not need the following imports here:
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.text import Text, Annotation
from matplotlib.patches import Polygon, Rectangle, Circle, Arrow
from matplotlib.widgets import SubplotTool, Button, Slider, Widget
from scipy.cluster.vq import vq, kmeans, whiten

from matplotlib.backends import pylab_setup
#############################################

# load 24bin data generated from Main10...
# get the average or not
def load_data(index, get_avg=False):
    hist_bin = np.loadtxt('hist/cell_{}.txt'.format(index))
    # print(len(hist_bin))
    hist_bin = np.reshape(hist_bin, (int(len(hist_bin)/24), 24))
    temp = hist_bin[0]
    if get_avg == True:
        for i in range(1, hist_bin.shape[0]):
            temp = temp + hist_bin[i]
        hist_bin = temp / hist_bin.shape[0]
        hist_bin = np.reshape(hist_bin, (1, 24))
    print('{} hist_bin shape {}'.format(index, hist_bin.shape))
    return hist_bin

def dimension_reduction(method, dimension):
    my_arrays = load_data(0, get_avg=True)
    for i in range(1, 89):
        my_arrays = np.concatenate((my_arrays, load_data(i, get_avg=True)), axis=0)
    # print(my_arrays.shape)
    my_tensor = torch.from_numpy(my_arrays)
    my_labels = np.loadtxt('hist/rotation.txt')
    print(my_labels.shape)
    # time.sleep(30)
    color_choice = ['b', 'g', 'r', 'yellow']
    # print(my_tensor)
    if method == 'tsne':
        module = manifold.TSNE(n_components=dimension,
                             init='random',
                             random_state=500,
                             early_exaggeration=5,
                             method='exact')
    elif method == 'PCA':
        module = PCA(n_components=3)
    x_numpy = my_tensor.data.cpu().numpy()
    x_reduced = module.fit_transform(x_numpy)
    print(x_reduced.shape)
    print(x_reduced)
    plt.figure()
    if dimension == 2:
        for i in range(x_reduced.shape[0]):
            color_index = int(my_labels[i])
            plt.plot(x_reduced[i, 0], x_reduced[i, 1], 'o', color=color_choice[color_index])
        plt.xticks()
        plt.yticks()

    elif dimension == 3:
        ax = plt.axes(projection='3d')
        for i in range(x_reduced.shape[0]):
            color_index = int(my_labels[i])
            ax.scatter(x_reduced[i, 0], x_reduced[i, 1], x_reduced[i, 2], color=color_choice[color_index])
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

    plt.savefig('plots/{}-24bin-{}d.pdf'.format(method, dimension))
    plt.show()



def feature_PCA():
    my_arrays = load_data(0, get_avg=True)
    for i in range(1, 89):
        my_arrays = np.concatenate((my_arrays, load_data(i, get_avg=True)), axis=0)
    my_tensor = torch.from_numpy(my_arrays)
    my_labels = np.loadtxt('hist/rotation.txt')
    color_choice = ['b', 'g', 'r', 'yellow']
    pca = PCA(n_components=3)
    x_numpy = my_tensor.data.cpu().numpy()
    x_pca = pca.fit_transform(x_numpy)

    plt.figure()
    ax = plt.axes(projection='3d')
    for i in range(x_pca.shape[0]):
        color_index = int(my_labels[i])
        # plt.plot(x_pca[i, 0], x_pca[i, 1], 'o', color=color_choice[color_index])
        ax.scatter(x_pca[i, 0], x_pca[i, 1], x_pca[i, 2], color=color_choice[color_index])
        # plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]),
        #          fontdict={'weight': 'bold', 'size': 9})
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

    # plt.xlim((-200, 0))
    # plt.ylim((-100, 100))
    # plt.xticks()
    # plt.yticks()
    # plt.savefig('tSNE-24bin.pdf')
    # plt.show()

def feature_visualization(method, dimension):
    color_choice = ['b', 'g', 'r', 'yellow']
    x_numpy = np.loadtxt('experiments/features.txt')
    my_labels = np.loadtxt('experiments/features_labels.txt')
    if method == 'tsne':
        module = manifold.TSNE(n_components=dimension,
                             init='pca',
                             random_state=500,
                             early_exaggeration=20,
                             method='exact')
    elif method == 'PCA':
        module = PCA(n_components=3)
    x_reduced = module.fit_transform(x_numpy)
    print(x_reduced.shape)
    print(x_reduced)
    plt.figure()
    if dimension == 2:
        for i in range(x_reduced.shape[0]):
            color_index = int(my_labels[i])
            if my_labels[i] == 0:
                status = 'Died'
            else:
                status = 'Survived'
            plt.plot(x_reduced[i, 0], x_reduced[i, 1], 'o', color=color_choice[color_index])
        plt.xticks()
        plt.yticks()

    elif dimension == 3:
        ax = plt.axes(projection='3d')
        for i in range(x_reduced.shape[0]):
            color_index = int(my_labels[i])
            ax.scatter(x_reduced[i, 0], x_reduced[i, 1], x_reduced[i, 2], color=color_choice[color_index])
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

    plt.legend(loc="lower right")
    plt.savefig('experiments/{}_{}.pdf'.format(method, dimension))
    plt.show()


def save_image(im, path):
    for i in range(im.shape[0]):
        for j in range(im[i].shape[0]):
#            image = Image.fromarray(im[i][j])
#            image = image.resize((224,224))
#            image = np.array(image)
            plt.figure(str(i)+'_'+str(j))
            plt.imshow(im[i][j],cmap='binary_r')
            plt.axis('off')
            plt.xticks([]),plt.yticks([]) #隐藏坐标线
            fig = plt.gcf()  
            dna1name='Conter'+str(i)+'_'+str(j)+'.png'
            fig.savefig(path+'IM/'+dna1name,dpi=100) 


def show_and_save_image():
    x1 = np.load('./ChaData/X1.npy')
    print(x1.shape)
    save_image(x1, path='./ChaData/')

    
    x2 = np.load('./MyData/X1.npy')
    print(x2.shape)
    save_image(x1, path='./MyData/')    





def feature_visualization2(method, dimension):
    color_choice = ['b', 'g', 'r', 'yellow']
    
    x1 = np.load('./ChaData/feature1.npy')
    print(x1.shape)
#    y1 = np.ones_like(x1)
    
    x2 = np.load('./MyData/feature1.npy')
    print(x2.shape)
#    y2 = np.zeros_like(x2)  
    
#    x2 = x2[100:x2.shape[0]]
    x_numpy = np.zeros((x1.shape[0]+x2.shape[0],4096))
    my_labels = np.zeros((x1.shape[0]+x2.shape[0]))
    
    x_numpy[0:x1.shape[0]] = x1
    x_numpy[x1.shape[0]:x1.shape[0]+x2.shape[0]] =x2
    
    my_labels[0:x1.shape[0]] = 1
    my_labels[x1.shape[0]:x1.shape[0]+x2.shape[0]] =0
#    x_numpy = np.load('./ChaData/feature.npy')
#    my_labels = np.load('./ChaData/label.npy')
    print('Size of data {}'.format(x_numpy.shape))    
    print('Size of labels {}'.format(my_labels.shape))
    died_index = np.where(my_labels == 0)
    survived_index = np.where(my_labels == 1)
    if method == 'tsne':
        module = manifold.TSNE(n_components=dimension,
                             init='pca',
                             random_state=100,
                             early_exaggeration=20,
                             method='exact')
    elif method == 'PCA':
        module = PCA(n_components=10)
    x_reduced = module.fit_transform(x_numpy)
    print(x_reduced)
    died_vector = x_reduced[died_index]
    survived_vector = x_reduced[survived_index]
    plt.figure()
    if dimension == 2:
        plt.plot(died_vector[:, 0], died_vector[:, 1],'ro',  label='Target Domain') #alpha=0.4

        plt.plot(survived_vector[:, 0], survived_vector[:, 1], 'b^', label='Source Domain') #, alpha=0.4
        # plt.plot(x_reduced[i, 0], x_reduced[i, 1], 'o', color=color_choice[color_index])
        plt.xticks()
        plt.yticks()
        

#        fig = plt.gcf()  
#        dna1name='tsne'+'.png'
#        fig.savefig(dna1name,dpi=100)

    elif dimension == 3:
        ax = plt.axes(projection='3d')
        ax.scatter(died_vector[:, 0], died_vector[:, 1], died_vector[:, 2], color='r', label='Target Domain')
        ax.scatter(survived_vector[:, 0], survived_vector[:, 1], survived_vector[:, 2], color='g', label='Source Domain')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

#    plt.title('t-SNE Visualization of Source And Target Domain Feature Vectors')
    plt.legend(loc="upper right")
    plt.savefig('tsne1.png',dpi=300)
    plt.show()
    
    
    
    
    
    
    
    return x_reduced

if __name__ == '__main__':

    xx=feature_visualization2(method='tsne', dimension=2)
    print(xx)
#    show_and_save_image()
    # feature_PCA()












