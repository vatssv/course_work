import argparse
import os
from PIL import Image
import glob
import shutil
import numpy as np
from scipy.stats import skew
from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage.color import label2rgb
#importing required libraries
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
import matplotlib.image as impg
from sklearn.metrics import euclidean_distances, pairwise
import math
from scipy.spatial import distance
from datetime import date, datetime

class Retrieval(object):
    k_images = []
    def __init__(self, *args, **kwargs):
        '''Fetch arguments from command line.'''
        self.args = args[0]
        self.model = self.args['model']
        self.image_id = self.args['image_id']
        self.k = int(self.args['k']) + 1
        self.db = self.args['image_db']
        self.work_dir = self.args['work_dir'] if self.args['work_dir'] else os.getcwd()

    def fetch_image_db(self):
        ''' To fetch the images from folder to match the test image against. '''
        self.image_list = []
        full_path = self.db + "*"
        for filename in glob.glob(full_path):
            im = Image.open(filename)
            im = np.asfarray(im)
            self.image_list.append([filename, im])

    def fetch_test_image(self):
        '''Fetch the test image to find similar images.'''
        image_path = self.db + self.image_id
        im = Image.open(image_path)
        self.test_image = np.asfarray(im)

    def write_features_to_file(self, features, method, feature_path):
        print('Storing features: ', feature_path)
        if os.path.exists(feature_path):
            shutil.rmtree(feature_path)
        os.makedirs(feature_path)
        features = features.tolist()
        file_template = f"{method}\n"
        for i in range(len(features)):
            file_template += str(features[i]) + '\n'
        output_path = feature_path + method + "_" + str(datetime.now()).replace(' ','-') + '.txt'
        with open(output_path, 'w+') as f:
            f.write(file_template)    

    def k_similar_images(self, id, distances):
        '''Given the distances, find k similar images from the folder.'''
        images = []
        for i in range(self.k):
            minpos = distances.index(min(distances))
            images.append(id[minpos])
            distances[minpos] = math.inf
        Retrieval.k_images = images[1:]
        return images[1:]

    def calculate_cm_similarity(self, db_image, test_image):
        '''Calculate similarity of color moments with Pearson's distance for the images when all models are combined.'''
        cm_image_features = np.asarray(self.calculate_cm_features(test_image))
        # print(f"Feature path is: {self.work_dir + '../Outputs_cm8x8/'}")
        self.write_features_to_file(cm_image_features, "cm8x8", self.work_dir + '../Outputs_cm8x8/')
        db_image_cm = np.asarray(self.calculate_cm_features(db_image))
        # print(f"Color moments feature for test image:\n Mean:\n {str(cm_image_features[0].tolist())}\n Standard Deviation:\n {str(cm_image_features[1].tolist())}\n Skewness\n {str(cm_image_features[2].tolist())}")
        return distance.correlation(db_image_cm.flatten(), cm_image_features.flatten())
    
    def calculate_cm_alone(self, db_image, test_image):
        '''Calculate similarity of color moments for the images with Manhattan Distance when it's compared alone.'''
        cm_image_features = np.asarray(self.calculate_cm_features(test_image))
        self.write_features_to_file(cm_image_features, "cm8x8", self.work_dir + '../Outputs_cm8x8/')
        db_image_cm = np.asarray(self.calculate_cm_features(db_image))
        # print(f"Color moments feature for test image:\n Mean:\n {str(cm_image_features[0].tolist())}\n Standard Deviation:\n {str(cm_image_features[1].tolist())}\n Skewness:\n {str(cm_image_features[2].tolist())}")
        return np.abs(db_image_cm-cm_image_features).sum()

    def calculate_combined_similarity(self, db_image, test_image):
        '''Calculate similarity of combined weighted features CM, ELBP and HOG and return the diatnces with Pearson's as similarity measure.'''
        d_hog = np.asarray(self.calculate_hog_similarity(db_image, test_image)) * 0.3
        # self.write_features_to_file(d_hog, "hog", self.work_dir + '../Outputs_hog/')
        d_elbp = self.calculate_elbp_similarity(db_image, test_image) * 0.2
        # self.write_features_to_file(d_elbp, "elbp", self.work_dir + '../Outputs_elbp/')
        d_cm = self.calculate_cm_alone(db_image, test_image) * 0.5
        # self.write_features_to_file(d_cm, "cm8x8", self.work_dir + '../Outputs_cm8x8/')
        d = d_hog + d_elbp + d_cm
        return d

    def calculate_elbp_similarity(self, db_image, test_image):
        '''Calculate similarity of ELBP features with Pearson's distance.'''
        elbp_image_features = np.asarray(self.calculate_elbp(test_image))
        self.write_features_to_file(elbp_image_features, "elbp", self.work_dir + '../Outputs_elbp/')
        db_image_elbp = np.asarray(self.calculate_elbp(db_image))
        # print(f"ELBP features for test image:\n {str(elbp_image_features.tolist())}")
        d = distance.correlation(db_image_elbp.flatten(), elbp_image_features.flatten())
        return d

    def calculate_hog_similarity(self, db_image, test_image):
        '''Calculate similarity of HOG features with Pearson's distance.'''
        hog_image_features = np.asarray(self.calculate_hog(test_image))
        self.write_features_to_file(hog_image_features, "hog", self.work_dir + '../Outputs_hog/')
        db_image_hog = np.asarray(self.calculate_hog(db_image))
        # print(f"HOG features for test image:\n {str(hog_image_features.tolist())}")
        d = distance.correlation(db_image_hog.flatten(), hog_image_features.flatten())
        return d

    @staticmethod
    def pre_process_for_cm(image):
        '''Divide the image into 8x8 blocks.'''
        M = 8
        N = 8
        v = image.copy()
        tiles = [v[x:x+M,y:y+N] for x in range(0,v.shape[0],M) for y in range(0,v.shape[1],N)]
        return tiles

    @staticmethod
    def calculate_elbp(image):
        '''Extract ELBP features from the image.'''
        return local_binary_pattern(image, 8, 1, method='ror').tolist()

    @staticmethod
    def calculate_hog(image):
        '''Extract HOG features from the image.'''
        fd = hog(image, orientations=9, pixels_per_cell=(8, 8), 
                        cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')
        truncatedFd=[float(str(i)[:12]) for i in fd[0].tolist()]
        return truncatedFd

    def calculate_cm_features(self, image):
        '''Extract color moments from the image.'''
        means,stds, skewness =[],[],[]
        tiles = self.pre_process_for_cm(image)
        for tile in tiles:
            means+=[tile.mean()]
            stds+=[np.std(tile)]
            skewness.append(skew(tile.flatten()))
        return [means, stds, skewness]

    def process_test_image(self):
        '''Process the arguments to run the required models and extract the required features.'''
        self.fetch_test_image()
        self.fetch_image_db()
        ids, distances = [], []

        for i in range(len(self.image_list)):
            if self.model == "hog":
                d = self.calculate_hog_similarity(self.image_list[i][1], self.test_image)
            elif self.model == "elbp":
                d = self.calculate_elbp_similarity(self.image_list[i][1], self.test_image)
            elif self.model == "cm8x8":
                d = self.calculate_cm_alone(self.image_list[i][1], self.test_image)
            else:
                d = self.calculate_combined_similarity(self.image_list[i][1], self.test_image)
            ids.append(self.image_list[i][0])
            distances.append(d)

        return self.k_similar_images(ids, distances)

def main():
    '''Fetch arguments from command line and save the resultant images.'''
    ap = argparse.ArgumentParser()
    ap.add_argument("image_db", help="Location of database of images.")
    ap.add_argument("image_id", help="Image id of test image (1-400).")
    ap.add_argument("model", nargs='?', help="Models to run with. 1. CM 2. ELBP 3. HOG", default=False)
    ap.add_argument("k", help="No. of images to retrieve matching the image.")
    ap.add_argument('work_dir', nargs='?', help="The working directory for the program.")
    args = vars(ap.parse_args())
    args['model'] = args['model'] if args['model'] in ["cm8x8", "elbp", "hog"] else '' 
    args['work_dir'] = args['work_dir'] if args['work_dir'] else str(os.getcwd()) + '/' 
    # args['work_dir'] = f"{args['work_dir']}/../Outputs_{args['model']}/"
    r = Retrieval(args)
    images = r.process_test_image()
    print(f'{len(images)} images retrieved.')
    print(f'Images are: {Retrieval.k_images}')

    #save the images in a directory called Outputs
    if args['model'] in ["cm8x8", "elbp", "hog"]:
        output_dir = f"{args['work_dir']}/../Outputs_{args['model']}/"
    else:
        output_dir = f"{args['work_dir']}/../Outputs_combined_models/"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

    for i in images:
        shutil.copy(i, output_dir)
        





if __name__ == "__main__":
    main()
