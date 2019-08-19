from skimage import transform
import warnings
import sys
from skimage import io
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from skimage import data, img_as_float
from skimage import exposure
import cv2
import pydicom
from glob import glob
import os
from os import path
import glob
import gdcm
import re
import warnings
import argparse
import sys
from tqdm import tqdm


class HistogramEqualization:

    def __init__(self):
        warnings.filterwarnings("ignore")
        self.dicom_path = ""
        self.window = 15000
        self.level = 6000
        self.cliplimit = .1
        self.bins = 1024
        self.option_export = False
        self.files = []
        self.index=[]

    matplotlib.rcParams['font.size'] = 8

    ####################################################################################
    ############################## CMD Options #########################################
    ####################################################################################

    def validate_args(self, string):
        print(string)
        if not (re.match(r'^(0?(\.\d+)?|1(\.0+)?)$', string)):
            raise argparse.ArgumentTypeError('Value has to be between 0 to 1')
        return string

        # Options parse
    def cmd_option_parser(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("-p", "--path", dest="path",
                            required=True, help="Dicom images folder path")
        parser.add_argument("-e", "--export",
                            action="store_true", dest="export", default=False,
                            help="export folder after processed images")
        parser.add_argument(
            "-w", help="window value in number", type=int, dest="window")
        parser.add_argument("-l", type=int, dest="level",
                            help="level value in number")
        parser.add_argument("-cl", type=self.validate_args,
                            dest="cliplimit", help="cliplimit between 0 to 1")
        parser.add_argument("-b", type=int, dest="bins",
                            help="bins value in number")
        parser.add_argument("-D", "-d", type=int, dest="index", nargs='+',
                            help="Disply image using the index,multiple index saparate by spaces")
        args = parser.parse_args()
        self.set_arguments(args)

    # set arguments
    def set_arguments(self, args):
        if(args.path is not None):
            self.dicom_path = args.path
            if not (os.path.isdir(self.dicom_path)):
                sys.stdout.write("Folder path does not exits! \n")
                sys.exit()

        if(args.export):
            self.option_export = True
        if(args.window is not None):
            self.winow = int(args.window)
        if(args.level is not None):
            self.level = int(args.level)
        if(args.cliplimit is not None):
            self.cliplimit = float(args.cliplimit)
        if(args.bins is not None):
            self.bins = int(args.bins)
        if(args.index is not None):
            self.index = list(set(args.index))

        print(args)

        # Start the processing....
        self.read_dicom_images(self.dicom_path)

    ####################################################################################
    ############################## Export Images #######################################
    ####################################################################################

    def export_image(self, ds=None, file=None, global_histogram=None, CLA_histogram=None):
        # decompress
        ds.decompress()

        if(self.dicom_path.endswith("/")):
            self.dicom_path = self.dicom_path[:-1]

        # # Export transformed images
        # self.create_dicom_from_numpy_transformed_equalized(
        #     ds=ds, file=file, global_histogram=global_histogram)

        # Export adaptive images
        self.create_png_from_numpy_adaptive_equalization(
            ds=ds, file=file, CLA_histogram=CLA_histogram)

    def create_dicom_from_numpy_transformed_equalized(self, ds=None, file=None, global_histogram=None):

        global_histogram = global_histogram*255
        for n, val in enumerate(ds.pixel_array.flat):
            ds.pixel_array.flat[n] = global_histogram.flat[n]

        ds.PixelData = ds.pixel_array.tobytes()
        new_file = file.split("/")
        try:
            os.mkdir(self.dicom_path+"_transformed_equalized")
        except Exception as e:
            pass

        ds.save_as(self.dicom_path+"_transformed_equalized" +
                   "/"+new_file[-1:][0])

    def create_dicom_from_numpy_adaptive_equalization(self, ds=None, file=None, CLA_histogram=None):

        CLA_histogram = CLA_histogram*255
        for n, val in enumerate(ds.pixel_array.flat):
            ds.pixel_array.flat[n] = CLA_histogram.flat[n]

        ds.PixelData = ds.pixel_array.tobytes()

        new_file = file.split("/")
        try:
            os.mkdir(self.dicom_path+"_adaptive_equalization")
        except Exception as e:
            pass

        ds.save_as(self.dicom_path+"_adaptive_equalization" +
                   "/"+new_file[-1:][0])


    def create_png_from_numpy_adaptive_equalization(self, ds=None, file=None, CLA_histogram=None):

        CLA_histogram = CLA_histogram*255

        # SLOW WAY TO COPY NUMPY MATRICES
        # for n, val in enumerate(ds.pixel_array.flat):
        #     ds.pixel_array.flat[n] = CLA_histogram.flat[n]

        from scipy import misc

        new_file = file.split("/")
        try:
            os.mkdir(self.dicom_path+"_adaptive_equalization")
        except Exception as e:
            pass

        misc.imsave(self.dicom_path+"_adaptive_equalization" +
                   "/"+new_file[-1:][0]+".png", CLA_histogram)
    ####################################################################################
    ############################## Display Plots #######################################
    ####################################################################################
    def plot_img_and_hist(self, image=None, axes=None, bins=255):

        image = img_as_float(image)
        ax_img, ax_hist = axes
        ax_cdf = ax_hist.twinx()
        # Display image
        ax_img.imshow(image, cmap=plt.cm.bone)
        # Display histogram
        try:
            ax_hist.hist(image.flatten(), bins=bins)
        except Exception as e:
            image = np.nan_to_num(image)
            ax_hist.hist(image.flatten(), bins=bins)
        return ax_img, ax_hist

    # Display the graph

    def display_graph(self, img, window_level_applied, global_histogram, CLA_histogram,):
        fig = plt.figure(figsize=(15, 15))
        axes = np.zeros((2, 4), dtype=np.object)
        axes[0, 0] = fig.add_subplot(2, 4, 1)

        for i in range(1, 4):
            axes[0, i] = fig.add_subplot(
                2, 4, 1+i, sharex=axes[0, 0], sharey=axes[0, 0])

        # add the subplots
        for i in range(0, 4):
            axes[1, i] = fig.add_subplot(2, 4, 5+i)

        ax_img, ax_hist = self.plot_img_and_hist(img, axes[:, 0])
        ax_img.set_title('Orignal Image')

        ax_img, ax_hist = self.plot_img_and_hist(
            window_level_applied, axes[:, 1])
        ax_img.set_title('Transformed images ')

        ax_img, ax_hist = self.plot_img_and_hist(
            global_histogram, axes[:, 2])
        ax_img.set_title('Transformed Equalization')
        ax_img, ax_hist = self.plot_img_and_hist(CLA_histogram, axes[:, 3])
        ax_img.set_title('Adaptive Equalization')

        # prevent overlap of y-axis labels
        fig.tight_layout()
        plt.show()

    ####################################################################################
    ############################## Process DICOM #######################################
    ####################################################################################

    # Apply window level
    def apply_window_level(self, data, window, level):
        offset = level - (window/2)
        newPixels = np.piecewise(data,
                                 [data < (level - (window/2)),
                                  data > (level + (window/2))],
                                 [0, 255, lambda data:
                                  ((data - offset)/window) * 255])
        return np.array(newPixels, dtype=np.uint8)

    # Apply Histogram on the images

    def apply_histogram_equalization(self, img=None, window_level_applied=None):
        if window_level_applied is not None and img is not None:

            global_histogram = exposure.equalize_adapthist(
                window_level_applied, clip_limit=self.cliplimit, nbins=self.bins)

            CLA_histogram = exposure.equalize_adapthist(
                img, clip_limit=self.cliplimit, nbins=self.bins)
            return global_histogram, CLA_histogram

    ####################################################################################
    ############################## Read DICOM #######################################
    ####################################################################################

    # read dicom folder
    def read_dicom_images(self, path=None):

        if path is not None and path != "":
            for file in glob.glob(path+"/*.dcm"):
                self.files.append(file)

        self.files = sorted(self.files)
        self.process_by_index()
    
    def process_dicom(self,index=None,display=False):

        self.ds = pydicom.dcmread(self.files[index-1])
        img = self.ds.pixel_array

        ii16 = np.iinfo(np.int16)
        self.pixels = np.multiply(ii16.max/np.max(img), img)

        self.window_level_applied = self.apply_window_level(
            self.pixels, self.window, self.level)

        global_histogram, CLA_histogram = self.apply_histogram_equalization(
            img=img, window_level_applied=self.window_level_applied)

        # Display Graph
        if(display):
            self.display_graph(img, self.window_level_applied,
                            global_histogram, CLA_histogram)

        if(self.option_export):
            # Save Process
            self.export_image(
                ds=self.ds, file=self.files[index-1], global_histogram=global_histogram, CLA_histogram=CLA_histogram)

    def process_by_index(self):
        
        if(len(self.index)>0):
            if(not max(self.index) > len(self.files)):
                for i in self.index:
                    self.process_dicom(index=i,display=True)
            else:
                sys.stdout.write("Index not found!\n")
            
        else:
            for i in tqdm(range(1, len(self.files))):
                self.process_dicom(index=i)

if __name__ == "__main__":
    hist_equlization = HistogramEqualization()
    hist_equlization.cmd_option_parser()
