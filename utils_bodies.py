import os
import re
from xml.etree import ElementTree, ElementInclude

import iptcinfo3
from iptcinfo3 import IPTCInfo
from PIL import Image
import cv2

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def get_meta_keys():

    """Returns the 'dict_keys' for the IPTCInfo"""
    
    dict_keys = ['object name', 
             'edit status', 
             'editorial update', 
             'urgency', 
             'subject reference', 
             'category', 
             'supplemental category', 
             'fixture identifier', 
             'keywords', 
             'content location code', 
             'content location name', 
             'release date', 
             'release time', 
             'expiration date', 
             'expiration time', 
             'special instructions', 
             'action advised', 
             'reference service', 
             'reference date', 
             'reference number', 
             'date created', 
             'time created', 
             'digital creation date', 
             'digital creation time', 
             'originating program', 
             'program version', 
             'object cycle', 
             'by-line', 
             'by-line title', 
             'city', 
             'sub-location', 
             'province/state', 
             'country/primary location code', 
             'country/primary location name', 
             'original transmission reference', 
             'headline', 
             'credit', 
             'source', 
             'copyright notice', 
             'contact', 
             'caption/abstract', 
             'local caption', 
             'writer/editor', 
             'image type', 
             'image orientation', 
             'language identifier', 
             'custom1', 
             'custom2', 
             'custom3', 
             'custom4', 
             'custom5', 
             'custom6', 
             'custom7', 
             'custom8', 
             'custom9', 
             'custom10', 
             'custom11', 
             'custom12', 
             'custom13', 
             'custom14', 
             'custom15', 
             'custom16', 
             'custom17', 
             'custom18', 
             'custom19', 
             'custom20']

    return(dict_keys)


def get_IPTC_data(path, filename):

    """Returns IPTC data given a path and a file name pertaining to one specific image."""

    file_path = os.path.join(path, filename)
    print(file_path)

    info = IPTCInfo(file_path, force=True)
    dict_keys = get_meta_keys()
    
    for i in dict_keys:
        if info[i] != None:
            if len(info[i]) > 0:
                print(f'key: {i}, info: {info[i]}\n')



def sample_rand_imgs(path, n = 5):

    """Samples n random images from path and print the image's IPTC data"""

    imgs = []
    for filename in os.listdir(path):
        if filename.split('.')[1] == 'jpg':
            imgs.append(filename)

    rand_idx = np.random.randint(0, len(imgs), n)
    rand_img_filename = np.array(imgs)[rand_idx].astype('str')

    for i in rand_img_filename:
        get_IPTC_data(path, i)


def get_df(path):

    """Returns a OD-df give a path to images/OD-xml """

    name = []
    xmin = []
    xmax = []
    ymin = []
    ymax = []
    width = []
    height = []
    depth = []
    sh = []
    filenames = []

    for filename in os.listdir(path):
        if filename.split('.')[1] == 'xml':

            file_path = os.path.join(path, filename)
                
            tree = ElementTree.parse(file_path)
                
            # Check if the file name in the xml matches thej actual file:
            filename_ = tree.findall('filename')[0].text
            if not filename_.split('.')[0] == filename.split('.')[0]:
                print(f'problem! {filename_.split(".")[0]} != {filename.split(".")[0]}')
                    #break

            # If it does; we go:
            else:

                lst_obj = tree.findall('object')
                lst_size = tree.findall('size')
                n_obj = len(lst_obj)

                for i in lst_obj:

                    name.append(i.find('name').text)
                    lst_box = i.findall('bndbox')

                    for j in lst_box:

                        xmin.append(j.find('xmin').text)
                        xmax.append(j.find('xmax').text)
                        ymin.append(j.find('ymin').text)
                        ymax.append(j.find('ymax').text)


                for k in lst_size:

                    width += [k.find('width').text] * n_obj
                    height += [k.find('height').text] * n_obj
                    depth += [k.find('depth').text] * n_obj
                    filenames += [filename_.split('.')[0]] * n_obj

                    #break # to only get one image

    columns = ['img_id', 'feature', 'xmin', 'xmax', 'ymin', 'ymax', 'width', 'height', 'depth']
    df = pd.DataFrame(list(zip(filenames, name, xmin, xmax, ymin, ymax, width, height, depth)), columns = columns)

    df['xmin'] = df['xmin'].astype('float64')
    df['xmax'] = df['xmax'].astype('float64')
    df['ymin'] = df['ymin'].astype('float64')
    df['ymax'] = df['ymax'].astype('float64')
    df['width'] = df['width'].astype('float64')
    df['height'] = df['height'].astype('float64')
    df['depth'] = df['depth'].astype('int')

    return(df)


def feature_dist_plot(df, show = False):

    """PLots and saves a distribution polt of the OD-features in the df"""

    features_count = df.groupby('feature').count().sort_values(['img_id'], ascending = True)[['img_id']].reset_index()

    plt.figure(figsize = [10,10])

    plt.title("Feature distribution\nJeppe's object detection, aug 2021", fontsize = 18)

    plt.barh(np.arange(0, features_count.shape[0],1) ,features_count['img_id'])
    plt.yticks(np.arange(0,features_count.shape[0],1), features_count['feature'], fontsize = 16)


    plt.savefig(f'feature_dist.pdf', bbox_inches="tight")   

    if show == True:
        plt.show()



def plt_obj(df_sub, image_i, feature):

    """df_sub is the random subset df given class_int.
    image_i is the i in the iteration. 
    classes in the list of classes. 
    Shape is img.shape.
    class_int is just the class in passed to the function above"""

    color = 'salmon'
    
    for _, row in df_sub[df_sub['img_id'] == image_i].iterrows():
        
        xmin =  row['xmin']
        xmax = row['xmax']
        ymin = row['ymin']
        ymax = row['ymax']

        xdiff = xmax - xmin
        ydiff = ymax - ymin

        xcenter = xmin + xdiff/2
        ycenter = ymin + ydiff/2


        plt.plot(xcenter, ycenter,'o', ms = 15, alpha = 0.5, color = color)

        plt.hlines(ycenter, xmin, xmax, color = color)
        plt.vlines(xcenter, ymin, ymax, color = color)

        plt.annotate(feature, [xmin, ymax], fontsize = 20, color = color)
                
        rect = Rectangle((xmin, ymin),xdiff,ydiff,linewidth = 2, edgecolor = color, facecolor='none')
        
        ax = plt.gca()
        ax.add_patch(rect)


def plot_img_subset(df, feature, show = False):

    """Plot a subset of images showing rectangles and labels of one given feature"""

    dir_images = '/home/simon/Documents/Bodies/data/jeppe/images'
    n = 9 # must be a square number: 1, 4, 9, 16, 25, 36, 49

    df_sub = df[df['feature'] == feature]
    img_subset = np.random.choice(df_sub['img_id'], n)

    plt.figure(figsize = [15,10])

    for i, j in enumerate(img_subset):
        path_image = f'{dir_images}/{j}.jpg'

        plt.subplot(np.sqrt(n), np.sqrt(n), i+1)
        plt.subplots_adjust(hspace = 0.2, wspace = 0.1)

        img = cv2.imread(path_image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # correcting the colors:
        plt.imshow(img)
        plt.title(j)

        suptitle = f'{feature}'
        plt.suptitle(suptitle, size=16)

        plt_obj(df_sub, j, feature)

    plt.savefig(f'{suptitle}.pdf', bbox_inches="tight")  

    if show == True:  
        plt.show()
