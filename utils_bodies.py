import os
import re
from xml.etree import ElementTree, ElementInclude
from collections import Counter

import iptcinfo3
from iptcinfo3 import IPTCInfo
from PIL import Image
import cv2

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import torch
import torchvision



# FUNCTIONS: ---------------------------------------------------------------------------------


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


def meta_to_df(df, path):

    """Returns new df with meta data from images. Needs old df and path to images"""

    df_expanded = df.copy()
    dict_keys = get_meta_keys()

    # create empty columns
    for k in dict_keys:    
        df_expanded[k] = np.nan #pd.NA #None

    # get IPCT info fro each img_id
    for i in df_expanded['img_id']:#[0:1]:

        #print(i) # for debug

        filename = i + '.jpg'
        file_path = os.path.join(path, filename)
        info = IPTCInfo(file_path, force=True)
        
        # Fill IPTC info into columns for i img_id
        for j in dict_keys:

            #print(j)  # for debug

            if info[j] != None:
                if len(info[j]) > 0:

                    #print(info[j])
                    #print(type(info[j]))
                    #print(len(info[j]))

                    if type(info[j]) == bytes:
                        # Just decode and add
                        df_expanded.loc[df_expanded['img_id'] == i, j] = info[j].decode('utf-8')

                    elif type(info[j]) == list:
                        # decode each entry in the list
                        temp_list = []
                        for n in info[j]:
                            temp_list.append(n.decode('utf-8'))

                        #print(temp_list)
                        # Make the list into a series of list fitting the size of the data frame slice
                        temp_list_series = pd.Series([temp_list] * df_expanded.loc[df_expanded['img_id'] == i, j].shape[0]) # it is a hack...
                        df_expanded.loc[df_expanded['img_id'] == i, j] = temp_list_series

                    else:
                        # just add
                        df_expanded.loc[df_expanded['img_id'] == i, j] = info[j]

    # remove columns with all NaNs
    df_cleaned = df_expanded.dropna(axis=1, how='all')
    return(df_cleaned)


def plot_boundry_boxes(dataset, images, targets, predictions):

    """Function for sanity check. Takes the dataset construted via the class MyDatast, 
    the images and targets extracted from said dataset ia dataloader, 
    and the predictions obtained via the Faster R-CNN. Returns/displays images with taget
    and predicted boundry boxes. 
    Right now the function still expect the coco classes for the predictions."""

    # dict for targets, int to str
    t_inst_classes = dataset.target_classes()
    inst_classes = dataset.coco_classes()

    for i,j in enumerate(images): # iterate over images i
        img = j[0].detach().numpy() # detach from device and tensor to numpy. j is now a touple with one entry... idk..
        img = img.squeeze() # remove batch dim so b,c,h,w -> c,h,w
        img = np.moveaxis(img, 0, -1) # move channels back so c,h,w -> h,w,c
        # note image is still range (0,1)

        threshold = .8 # needs to be and input to the function
        color1 = 'orange' # needs to vary according to class
        color2 = 'salmon' # needs to vary according to class


        n_obj = predictions[i]['boxes'].shape[0]

        #plot prediction boundry boxes
        for k in range(n_obj): # iterate over objects j in image i
            
            if predictions[i]['scores'][k] > threshold: # is the predictions clear the threshold

                box = predictions[i]['boxes'][k].detach().numpy() # boxes in xmin, ymin, xmax, ymax format

                xmin = box[0]
                ymin = box[1]
                xmax = box[2]
                ymax = box[3]

                xdiff = xmax - xmin
                ydiff = ymax - ymin

                plt.annotate(inst_classes[predictions[i]['labels'][k]], [xmin, ymax], fontsize = 20, color = color1) 
                # you use inst_classes here!!! that list needs to be loaded from somewhere..
                            
                rect = Rectangle((xmin, ymin),xdiff,ydiff,linewidth = 2, edgecolor = color1, facecolor='none')
                    
                ax = plt.gca()
                ax.add_patch(rect)

        #plot target boundry boxes. if-statment handelse images with only one box = one less dim.
        if len(targets[i]['boxes'].shape) == 3:
            t_obj = targets[i]['boxes'].squeeze()
            t_label = targets[i]['labels'].squeeze()

        elif len(targets[i]['boxes'].shape) == 2:
            t_obj = targets[i]['boxes']
            t_label = targets[i]['labels']

        else:
            print('wrong dims...')

        for m, l in enumerate(t_obj): # for target l in images i. and squeeze to remove batch dim
                
                t_box = l.detach().numpy() # boxes in xmin, ymin, xmax, ymax format

                t_xmin = t_box[0]
                t_ymin = t_box[1]
                t_xmax = t_box[2]
                t_ymax = t_box[3]

                t_xdiff = t_xmax - t_xmin
                t_ydiff = t_ymax - t_ymin

                plt.annotate(t_inst_classes[t_label[m].item()], [t_xmin, t_ymax], fontsize = 20, color = color2)
                            
                t_rect = Rectangle((t_xmin, t_ymin), t_xdiff, t_ydiff, linewidth = 2, edgecolor = color2, facecolor='none')
                    
                ax = plt.gca()
                ax.add_patch(t_rect)

        # plot image 
        plt.imshow(img)
        plt.title('test')

        plt.show()



# CLASSES: ---------------------------------------------------------------------------------
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root = '/home/simon/Documents/Bodies/data/jeppe/', transforms = None, n_obs = 100):
        self.root = root
        self.transforms = transforms
        self.n_obs = n_obs

        # the selection need to happen here
        self.classes = [''] + self.__get_classes__() # list of classes accroding to n_obs, see __get_classes__
        self.classes_int = np.arange(0,len(self.classes)) # from 1 since no background '0'
        self.boxes = self.__get_boxes__() # list of xml files (box info) to n_obs, see __get_classes__
        self.imgs = [f"{i.split('.')[0]}.jpg" for i in self.boxes] # list of images - only take images with box info! and > n_obs
             
    def __get_classes__(self):
        """Creates a list of classes with >= n_obs observations"""
        n_obs = self.n_obs
        path = os.path.join(self.root, "images")

        obj_name = []
        classes = []

        # Get all objects that have been annotated
        for filename in os.listdir(path):
            if filename.split('.')[1] == 'xml':
                box_path = os.path.join(path, filename)

                tree = ElementTree.parse(box_path)
                lst_obj = tree.findall('object')

                for j in lst_obj:
                    obj_name.append(j.find('name').text)


        # now, only keep the objects w/ >= n_obs observations
        c = Counter(obj_name)

        for i in c.items():
            if i[1] >= n_obs:
                classes.append(i[0])
        
        return(classes)

    def __get_boxes__(self):
        """Make sure you only get images with valid boxes frrom the classes list - see __get_classes__"""

        path = os.path.join(self.root, "images")

        boxes = []
        # Get all objects that have been annotated
        for filename in os.listdir(path):
            if filename.split('.')[1] == 'xml':
                box_path = os.path.join(path, filename)

                tree = ElementTree.parse(box_path)
                lst_obj = tree.findall('object')

                # If there is one or more objects from the classes list, save the box filename
                if len(set([j.find('name').text for j in lst_obj]) & set(self.classes)) > 0:
                    boxes.append(filename)

        # Sort and return the boxes
        boxes = sorted(boxes)
        return(boxes)

    def __getitem__(self, idx):
        # dict to convert classes into classes_int
        class_to_int = dict(zip(self.classes,self.classes_int))        

        # load images
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        box_path = os.path.join(self.root, "images", self.boxes[idx])
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize img 800x800 --------------------------------------------
        target_size = 800

        y_orig_size = img.shape[0] # the original y shape
        x_orig_size = img.shape[1] # the original x shape
        y_scale = target_size/y_orig_size # scale factor for boxes
        x_scale = target_size/x_orig_size # scale factor for boxes

        img = cv2.resize(img, (target_size, target_size))
        # ----------------------------------------------------------------

        img = np.moveaxis(img, -1, 0) # move channels in front so h,w,c -> c,h,w
        img = img / 255.0 # norm ot range 0-1. Might move out..
        img = torch.Tensor(img)

        # Open xml path 
        tree = ElementTree.parse(box_path)

        lst_obj = tree.findall('object')

        obj_name = []
        obj_ids = []
        boxes = []

        for i in lst_obj:
        # here you need to ignore classes w/ n > n_obs

            obj_name_str = i.find('name').text
            if obj_name_str in self.classes:

                obj_name.append(obj_name_str) # get the actual class name
                obj_ids.append(class_to_int[i.find('name').text]) # get the int associated with the class name
                lst_box = i.findall('bndbox')

                for j in lst_box:

                    xmin = float(j.find('xmin').text) * x_scale # scale factor to fit resized image
                    xmax = float(j.find('xmax').text) * x_scale
                    ymin = float(j.find('ymin').text) * y_scale
                    ymax = float(j.find('ymax').text) * y_scale
                    boxes.append([xmin, ymin, xmax, ymax])
            else:
                pass

        num_objs = len(obj_ids) # number of objects

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(obj_ids, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes 
        target["labels"] = labels
        target["image_id"] = image_id 
        target["area"] = area
        target["iscrowd"] = iscrowd 

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs) # right now you do not differentiate between annotated images and not annotated images... 


    def target_classes(self):
        t_inst_classes = dict(zip(self.classes_int,self.classes)) # just a int to string dict
        return(t_inst_classes)

    def coco_classes(self):
        inst_classes = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'] # a "ordered" list of the coco categories
        return(inst_classes) 



# OLD ---------------------------------------------------------------------------------------------------
class MyDataset2(torch.utils.data.Dataset):
    def __init__(self, root = '/home/simon/Documents/Bodies/data/jeppe/', transforms = None):
        self.root = root
        self.transforms = transforms
        self.boxes = [i for i in list(sorted(os.listdir(os.path.join(root, "images")))) if str(i).split('.')[1] == 'xml'] # list of xml files (box info)
        self.imgs = [f"{i.split('.')[0]}.jpg" for i in self.boxes] # list of images - only take images with box info!
        #self.classes = open(os.path.join(root, "images/classes.txt"),"r").read().split('\n')[0:-1] # the classes from the classes.txt file
        # self.classes = ['background'] + self.__get_classes__()
        self.classes = [''] + self.__get_classes__() # 0 index reserved for background

        self.classes_int = np.arange(0,len(self.classes)) # from 1 since no background '0'

    def __get_classes__(self, n_obs = 150):
        """Creates a list of classes with >= n_obs observations"""

        path = os.path.join(self.root, "images")

        obj_name = []
        classes = []

        # Get all objects that have been annotated
        for filename in os.listdir(path):
            if filename.split('.')[1] == 'xml':
                box_path = os.path.join(path, filename)

                tree = ElementTree.parse(box_path)
                lst_obj = tree.findall('object')

                for j in lst_obj:
                    obj_name.append(j.find('name').text)


        # now, only keep the objects w/ >= n_obs observations
        c = Counter(obj_name)

        for i in c.items():
            if i[1] >= n_obs:
                classes.append(i[0])
        
        return(classes)

    def __getitem__(self, idx):
        # dict to convert classes into classes_int
        class_to_int = dict(zip(self.classes,self.classes_int))        

        # load images
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        box_path = os.path.join(self.root, "images", self.boxes[idx])
        
        #img = Image.open(img_path).convert("RGB") # maybe you also need the dim and norm stuff here.
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize img 800x800 --------------------------------------------
        target_size = 800

        y_orig_size = img.shape[0] # the original y shape (indx before you move channels )
        x_orig_size = img.shape[1] # the original x shape (indx before you move channels )
        y_scale = target_size/y_orig_size # scale factor for boxes
        x_scale = target_size/x_orig_size # scale factor for boxes

        img = cv2.resize(img, (target_size, target_size))
        # ----------------------------------------------------------------

        img = np.moveaxis(img, -1, 0) # move channels in front so h,w,c -> c,h,w
        img = img / 255.0 # norm ot range 0-1. Might move out..
        img = torch.Tensor(img)

        # Open xml path 
        tree = ElementTree.parse(box_path)

        lst_obj = tree.findall('object')
        # lst_size = tree.findall('size') # are you using this?

        obj_name = []
        obj_ids = []
        boxes = []

        for i in lst_obj:

            obj_name_str = i.find('name').text
            if obj_name_str in self.classes: # only keep the object if it is in your list of classes. See __get_classes__

                obj_name.append(obj_name_str) # get the actual class name
                obj_ids.append(class_to_int[i.find('name').text]) # get the int associated with the class name
                lst_box = i.findall('bndbox')

                for j in lst_box:

                    xmin = float(j.find('xmin').text) * x_scale # scale factor to fit resized image
                    xmax = float(j.find('xmax').text) * x_scale
                    ymin = float(j.find('ymin').text) * y_scale
                    ymax = float(j.find('ymax').text) * y_scale
                    boxes.append([xmin, ymin, xmax, ymax])
            
            else:
                pass

        num_objs = len(obj_ids) # number of objects

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(obj_ids, dtype=torch.int64)

        image_id = torch.tensor([idx])

        # temp selution to when all the classes are droped for being too rare. So boxes is empty list...
        if len(torch.as_tensor(boxes, dtype=torch.float32)) == 0:
            area = 0

        else:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # ------------------------------------------------------------------------------------------------

        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Idk i squeeze helps or is warrented here...
        target = {}
        target["boxes"] = boxes #.view(-1,4)
        target["labels"] = labels #.squeeze()
        target["image_id"] = image_id #.squeeze()
        target["area"] = area #.squeeze()
        target["iscrowd"] = iscrowd #.squeeze()

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target


    def __len__(self):
        return len(self.imgs) # right now you do not differentiate between annotated images and not annotated images... 


    def target_classes(self):
        t_inst_classes = dict(zip(self.classes_int,self.classes)) # just a int to string dict
        return(t_inst_classes)

    def coco_classes(self):
        inst_classes = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'] # a "ordered" list of the coco categories
        return(inst_classes) 

class MyDataset1(torch.utils.data.Dataset):
    def __init__(self, root = '/home/simon/Documents/Bodies/data/jeppe/', transforms = None):
        self.root = root
        self.transforms = transforms
        self.boxes = [i for i in list(sorted(os.listdir(os.path.join(root, "images")))) if str(i).split('.')[1] == 'xml'] # list of xml files (box info)
        self.imgs = [f"{i.split('.')[0]}.jpg" for i in self.boxes] # list of images - only take images with box info!
        #self.imgs = [i for i in list(sorted(os.listdir(os.path.join(root, "images")))) if str(i).split('.')[1] == 'jpg'] # list of images
        self.classes = open(os.path.join(root, "images/classes.txt"),"r").read().split('\n')[0:-1] # the classes from the classes.txt file
        self.classes_int = np.arange(1,len(self.classes)+1) # from 1 since no background '0'

    def __getitem__(self, idx):
        # dict to convert classes into classes_int
        class_to_int = dict(zip(self.classes,self.classes_int))        

        # load images
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        box_path = os.path.join(self.root, "images", self.boxes[idx])
        
        #img = Image.open(img_path).convert("RGB") # maybe you also need the dim and norm stuff here.
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.moveaxis(img, -1, 0) # move channels in front so h,w,c -> c,h,w
        img = img / 255.0 # norm ot range 0-1. Might move out..
        img = torch.Tensor(img)

        # Open xml path 
        tree = ElementTree.parse(box_path)

        lst_obj = tree.findall('object')
        lst_size = tree.findall('size') # are you using this?
        num_objs = len(lst_obj) # number of objects

        obj_name = []
        obj_ids = []
        boxes = []

        for i in lst_obj:

            obj_name.append(i.find('name').text) # get the actual class name
            obj_ids.append(class_to_int[i.find('name').text]) # get the int associated with the class name
            lst_box = i.findall('bndbox')

            for j in lst_box:

                xmin = float(j.find('xmin').text)
                xmax = float(j.find('xmax').text)
                ymin = float(j.find('ymin').text)
                ymax = float(j.find('ymax').text)
                boxes.append([xmin, ymin, xmax, ymax])
                # boxes.extend([xmin, ymin, xmax, ymax])


        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(obj_ids, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Idk i squeeze helps or is warrented here...
        target = {}
        #target["boxes"] = boxes.squeeze()
        target["boxes"] = boxes #.view(-1,4)
        target["labels"] = labels #.squeeze()
        target["image_id"] = image_id #.squeeze()
        target["area"] = area #.squeeze()
        target["iscrowd"] = iscrowd #.squeeze()

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target


    def __len__(self):
        return len(self.imgs) # right now you do not differentiate between annotated images and not annotated images... 


    def target_classes(self):
        t_inst_classes = dict(zip(self.classes_int,self.classes)) # just a int to string dict
        return(t_inst_classes)

    def coco_classes(self):
        inst_classes = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'] # a "ordered" list of the coco categories
        return(inst_classes) 

