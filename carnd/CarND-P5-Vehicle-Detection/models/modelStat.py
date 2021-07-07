'''
NOTE: The filepaths used in this script are / delimited for use on a linux computer
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import glob
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib
from sklearn.utils import shuffle
from scipy.ndimage.measurements import label
from skimage.feature import hog
import time
from moviepy.editor import VideoFileClip
from IPython.display import HTML


def plot_image(img_list, gray_maps, titles=None, axes_off=True, imsize=(20,20), bgr=None):
    '''
    Utility for examining results from project functions. Plots images
    Input: img_list = List of images
           gray_maps = List of boolans indicating which of img_list is grayscale
           titles = List of strings indicating titles of each image
           axes_off = Boolean flag indicating if plot axes should be hidden
           imsize = Int tuple of image size
           bgr = List of booleans indicating which of img_list is in bgr
    Output: ax = axis object from plot
    '''
    if bgr == None:
        bgr = [False for i in range(len(img_list))]
    
    if len(img_list)==1:
        f, ax = plt.subplots(1, 1, figsize=imsize)
        if gray_maps[0]:
            ax.imshow(img_list[0], cmap='gray')
        elif bgr:
            ax.imshow(img_list[0][:,:,::-1])
        else:
            ax.imshow(img_list[0])
        if axes_off:
            ax.axis('off')
        if titles != None:
            ax.set_title(titles[0], fontsize=20)
    else:
        f, ax = plt.subplots(1, len(img_list), figsize=imsize)
        f.tight_layout
        ax = ax.ravel()
        for i in range(len(img_list)):
            if gray_maps[i]:
                ax[i].imshow(img_list[i], cmap='gray')
            elif bgr[i]:
                ax[i].imshow(img_list[i][:,:,::-1])
            else:
                ax[i].imshow(img_list[i])
            if axes_off:
                ax[i].axis('off')
            if titles != None:
                ax[i].set_title(titles[i], fontsize=20)
    return ax

	
def color_hist(img, nbins=32, bins_range=(0, 256)):
    '''
    Create a histogram of values for each color channel in an image
    Inputs: img = Image
            nbins = Number of bins to sort values into
            bins_range = Range of values image pixels may have
    Output: Array of histogram values
    '''
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    
    # Concatenate the histograms into a single feature vector and return
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    
    return hist_features
	

def bin_spatial(img, size=(32, 32)):
    '''
    Bin the pixels of an image into those of a smaller image using cv2's default bilinear interpolation
    Inputs: img = Image
            size = 2-tuple of binned image dimensions
    Output: Flattened array of pixel values from binned image
    '''
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    
    return features

def convert_color(img, conv='BGR2YCrCb'):
    return cv2.cvtColor(img, eval('cv2.COLOR_'+conv))



def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    '''
    Shorthand for producing a histogram-oriented graph (HOG) for an image
    Inputs: img = Image
            orient = Number of gradient orientations to bin values into
            pix_per_cell = Number of pixels per cell
            cell_per_block = Number of cells per block
            vis = Boolean to toggle return of an image of the graph
            feature_vec = Boolean to indicate whether this computation is meant to return a single feature vector
    Output: features (if vis) = HOG values
            hog_image (if vis) = Image of HOG
            output (if not vis) = HOG values
    '''
    # Comput the HOG with the given parameters
    # Uses 'L2-Hys' for the normalization and sets transform_sqrt to True
    output = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                 cells_per_block=(cell_per_block, cell_per_block), block_norm= 'L2-Hys',
                 transform_sqrt=True, visualise=vis, feature_vector=feature_vec)

    # Return two outputs if vis == True
    if vis == True:
        features, hog_image = output[0], output[1]
        return features, hog_image
    # Otherwise return one output
    else:
        return output

def extract_features(imgs, color_space='BGR',
                     spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256),
                     orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    '''
    Extract the HOG, spatial bin, and/or color histogram features from a list of images
    Inputs: imgs = List of image file-paths
            color_space = String identifying the color space from which the features should be extracted
            spatial_size = 2-Tuple indicating bin counts in x and y respectively
            hist_bins = Number of bins into which color values should be binned
            hist_range = Range of values image pixels may have
            orient = Number of gradient orientations to bin values into
            pix_per_cell = Number of pixels per cell
            cell_per_block = Number of cells per block
            hog_channel = Image color-channel(s) to use for computing HOG (Pass string 'ALL' to use all channels)
            spatial_feat = Boolean indicating whether or not spatial features should be extracted
            hist_feat = Boolean indicating whether or not color features should be extracted
            hog_feat = Boolean indicating whether or not HOG features should be computed
    Outputs: List of each image's feature vector
    '''
    
    # Create a list to append feature vectors to
    features = []
    
    # Iterate through the list of images
    for file in imgs:
        
        file_features = []
        
        # Read in the image
        image = cv2.imread(file)
        
        # Apply color conversion if necessary
        if color_space != 'BGR':
            feature_image = convert_color(image, 'BGR2'+color_space)
        else:
            feature_image = np.copy(image)   
        
        # Get spatail features
        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        
        # Get color features
        if hist_feat == True:
            hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_bins)
            file_features.append(hist_features)
            
        # Get HOG features
        if hog_feat == True:
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
                
            file_features.append(hog_features)
        
        # Add this image's features to the list
        features.append(np.concatenate(file_features))

    return features


def model_measure(clfname, cars, notcars, cols=[], keycols=[], cspace='BGR',
                  hog_channel='ALL', orient=9, pix_per_cell=8, cell_per_block=2,
                  spatial_params=8, hist_params=16,
                  spatial_feat=True, hist_feat=True, hog_feat=True):
    
    test_params = dict.fromkeys(cols)
    
    colorspace = cspace
    spatial_size = (spatial_params, spatial_params)
    hist_bins = hist_params
    y_start_stop = [400, None]
    
    print('Extracting features...')
    # Extract features
    t0 = time.time()
    car_features = extract_features(cars, color_space=colorspace, orient=orient,
                                    pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_size=spatial_size,
                                    hist_bins=hist_bins, hist_feat=hist_feat,
                                    spatial_feat=spatial_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=colorspace, orient=orient,
                                       pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_size=spatial_size,
                                       hist_bins=hist_bins, hist_feat=hist_feat,
                                       spatial_feat=spatial_feat, hog_feat=hog_feat)
    t1 = time.time()
    print('%.3f seconds to extract features.\n' % (t1-t0))
    test_params[cols[0]] = round(t1-t0, 3)
    
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    
    # Shuffle and split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X, y = shuffle(X, y, random_state=rand_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)

    t0 = time.time()
    # Fit a per-column scaler
    print('Fitting scaler...')
    X_scaler = StandardScaler().fit(X_train)
    t1 = time.time()
    
    print('%.3f seconds to fit scaler.\n' % (t1-t0))
    test_params[cols[1]] = round(t1-t0, 3)
    
    # Save the scaler for use in other kernels
    print('Saving scaler...')
    joblib.dump(X_scaler, clfname+'_scaler.p')
    print('Scaler saved.')

    # Apply the scaler to X
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)
    
    
    # Use a linear SVC
    svc = LinearSVC()

    # Check the training time for the SVC
    print('Training classifier...')
    t0 = time.time()
    svc.fit(X_train, y_train)
    t1 = time.time()
    print('%.3f seconds to train classifier.\n' % (t1-t0))
    test_params[cols[2]] = round(t1-t0, 3)

    

    # Check the prediction time for a single sample
    print('Testing model...')
    t0 = time.time()
    pred = svc.predict(X_test)
    t1 = time.time()
    print('%.3f seconds to predict test classes\n' % (t1-t0))
    test_params[cols[3]] = round(t1-t0, 3)
    
    # Check the score of the SVC
    print('Measuring model accuracy...')
    acc = svc.score(X_test, y_test)
    print('%.3f = Test Accuracy of SVC\n' % acc)
    test_params[cols[4]] = round(acc, 3)

    # Save the classifier for use in other kernels
    print('Saving SVC...')
    joblib.dump(svc, clfname+'_clf.p')
    print('Model saved.')
    
    # Record parameters used to perform the preceding and return
    for i in range(len(keycols)):
        test_params[keycols[i]] = [cspace, hog_channel, orient, pix_per_cell, cell_per_block][i]
    
    return test_params



#################################################################################################################################################################################################


channels = ['ALL']
orients = [6, 9, 12]
ppcs = [8, 16]
cbps = [2, 4]
cspaces = ['HSV', 'LUV', 'HLS', 'YUV', 'YCrCb']

cars = glob.glob('vehicles/*/*.png', recursive=True)
nocars = glob.glob('non-vehicles/*/*.png', recursive=True)

cols = ['t_featureExtract','t_fitScaler','t_trainClassifier','t_predictTestData','testAccuracy']
keycols = ['cspace', 'hog_channel', 'orient', 'pix_per_cell', 'cell_per_block']
results = pd.DataFrame(columns=keycols+cols)


count = 0
for cspace in cspaces:
    for channel in channels:
        for orient in orients:
            for ppc in ppcs:
                for cbp in cbps:
                    r = model_measure(str(count), cars, nocars, cols, keycols, cspace=cspace,
                                      hog_channel=channel, orient=orient, pix_per_cell=ppc, cell_per_block=cbp,
                                      spatial_feat=False, hist_feat=False, hog_feat=True)
                    count+=1
                    results = results.append(r, ignore_index=True)
                    results.to_csv('./modelSelection.csv')                        
                        
results.to_csv('./modelSelection.csv')



#################################################################################################################################################################################################
