# LIBRAIRIES
from skimage.color import rgb2hsv, rgb2gray
from skimage.morphology import closing, opening, disk, remove_small_holes, remove_small_objects, binary_dilation, erosion
from skimage.transform import rotate, resize, AffineTransform, warp
from sklearn.metrics.pairwise import euclidean_distances
from skimage.measure import regionprops
import matplotlib.image as mpimg


import os
import copy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from src.Project_utils import *

from sklearn.metrics import accuracy_score, f1_score
import cv2


'''
def find_contour(images: np.ndarray): #from iapr lab2
    """
    Find the contours for the set of images
    
    Args
    ----
    images: np.ndarray (N, 28, 28)
        Source images to process

    Return
    ------
    contours: list of np.ndarray
        List of N arrays containing the coordinates of the contour. Each element of the 
        list is an array of 2d coordinates (K, 2) where K depends on the number of elements 
        that form the contour. 
    """

    # Get number of images to process
    N, _, _ = np.shape(images)
    # Fill in dummy values (fake points)
    contours = [np.array([[0, 0], [1, 1]]) for i in range(N)]

    for i in range(N):
        contours[i], _ = cv2.findContours(images[i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours[i] = contours[i][0][:, 0, :]

    return contours



def display_img_with_contour(
    image: np.ndarray,
    label: str,
    title: str,
    cnt: np.ndarray = None):

    # Create ONE large figure
    fig, axis = plt.subplots(figsize=(8, 8))

    # Show image
    axis.imshow(image, interpolation="nearest")

    # Remove axes
    axis.axis("off")

    # Small title above image
    axis.set_title(label, fontsize=16)

    # Draw contour
    if cnt is not None and len(cnt) > 0:
        axis.plot(cnt[:, 0], cnt[:, 1], 'r-', linewidth=2)

    # Main title
    fig.suptitle(title, fontsize=20)

    plt.tight_layout()
    #plt.show()
    #remove to see figure
    plt.close()
    
    
def extract_hsv_channels(img):

    #Extract HSV channels from the input image.

    # Get the shape of the input image
    M, N, C = np.shape(img)

    # Define default values for HSV channels
    data_h = np.zeros((M, N))
    data_s = np.zeros((M, N))
    data_v = np.zeros((M, N))

    data_hsv = rgb2hsv(img)
    data_h = data_hsv[:, :, 0]
    data_s = data_hsv[:, :, 1]
    data_v = data_hsv[:, :, 2]

    return data_h, data_s, data_v


def apply_closing(img_th, disk_size):

    img_closing = np.zeros_like(img_th)
    img_closing = closing(img_th, disk(disk_size))

    return img_closing

def apply_opening(img_th, disk_size):

    img_opening = np.zeros_like(img_th)

    img_opening = opening(img_th, disk(disk_size))

    return img_opening

def apply_erosion(img_th, disk_size):

    img_erosion = np.zeros_like(img_th)

    img_erosion = erosion(img_th, footprint=disk(disk_size))

    return img_erosion



def apply_hsv_threshold_v2(img, mode="red"):

    M, N, C = np.shape(img)

    data_h, data_s, data_v = extract_hsv_channels(img=img)

    if mode == "red":
        h_min1, h_max1 = 0.0, 0.05 # car 2 pics
        h_min2, h_max2 = 0.95, 1.0
        s_min, s_max = 0.4, 1.0
        v_min, v_max = 0.3, 1.0
        img_th = (
            (((data_h >= h_min1) & (data_h <= h_max1)) | 
             ((data_h >= h_min2) & (data_h <= h_max2))) &
            (data_s > s_min) & (data_s < s_max) &
            (data_v > v_min) & (data_v < v_max)
        )
        return img_th

    if mode == "blue":
        h_min, h_max = 0.52, 0.68
        s_min, s_max = 0.3, 1.0
        v_min, v_max = 0.3, 1.0

    if mode == "yellow":
        h_min, h_max = 0.06, 0.20
        s_min, s_max = 0.2, 1.0
        v_min, v_max = 0.8, 1.0

    if mode == "green":
        # Vert : Hue ~0.25-0.45
        h_min, h_max = 0.22, 0.48
        s_min, s_max = 0.3, 1.0
        v_min, v_max = 0.3, 1.0

    if mode == "black":
        h_min, h_max = 0.0, 1.0  
        s_min, s_max = 0.0, 1.0 
        v_min, v_max = 0.0, 0.50  

    img_th = (
        (data_h > h_min) & (data_h < h_max) &
        (data_s > s_min) & (data_s < s_max) &
        (data_v > v_min) & (data_v < v_max)
    )

    return img_th

def find_area_by_type(
    img: np.ndarray,
    mode: str = "red",
    visualize_hsv: bool = False,
    visualize_mask: bool = False,
    disk_size: int = 20,
    object_min_size: int = 30,
):

    if visualize_hsv:
        plot_colors_histo(
            img=img,
            func=extract_hsv_channels,
            labels=["Hue", "Saturation", "Value"],
        )

    img_th = apply_hsv_threshold_v2(img, mode=mode)

    if visualize_mask:
        plot_thresholded_image(
            img=img,
            func=lambda img: apply_hsv_threshold_v2(img, mode=mode),
            title=f"{mode.capitalize()} detection in HSV space"
        )

    out_mask = apply_opening(img_th, disk_size)
    out_mask = apply_closing(out_mask, disk_size)
    


    #out_mask = remove_small_objects(out_mask, min_size=object_min_size)


    return out_mask



def combined_mask(img, color_masks):

    combined_mask = np.zeros_like(list(color_masks.values())[0])
    for mask in color_masks.values():
        combined_mask = combined_mask | mask
    return combined_mask

'''  

def linear_interpolation(contours: np.ndarray, n_samples: int = 11):
    """
    Perform interpolation/resampling of the contour across n_samples.
    
    Args
    ----
    contours: list of np.ndarray
        List of N arrays containing the coordinates of the contour. Each element of the 
        list is an array of 2d coordinates (K, 2) where K depends on the number of elements 
        that form the contour. 
    n_samples: int
        Number of samples to consider along the contour.

    Return
    ------
    contours_inter: np.ndarray (N, n_samples, 2)
        Interpolated contour with n_samples
    """

    N = len(contours)
    contours_inter = np.zeros((N, n_samples, 2))
    
    # ------------------
    # Your code here ... 
    # ------------------
    for i in range(N):
        contour = contours[i]
        length = len(contour)
        if length >= n_samples:
            indices = np.linspace(0, length - 1, n_samples).astype(int)
            contours_inter[i] = contour[indices]
        else:
            indices = np.arange(length)
            contours_inter[i, :length] = contour[indices]
            contours_inter[i, length:] = contour[-1]
        
    return contours_inter



#this function was 100% vibe coded
def plot_interpolated_contours(
    mask: np.ndarray,
    contours_inter: np.ndarray,
    color: str = "red",
    point_size: int = 10,
    linewidth: int = 2,
    path = ""
):
    """
    Display interpolated contours on top of a mask image.

    Args
    ----
    mask: np.ndarray (H, W)
        Binary/grayscale mask image.

    contours_inter: np.ndarray (N, n_samples, 2)
        Interpolated contours.

    color: str
        Color used for contour visualization.

    point_size: int
        Size of sampled contour points.

    linewidth: int
        Width of contour lines.
    """

    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(10, 10))

    # Show mask
    plt.imshow(mask, cmap="gray")

    # Plot each contour
    for contour in contours_inter:

        # x/y coordinates
        x = contour[:, 0]
        y = contour[:, 1]

        # Draw contour line
        plt.plot(
            x,
            y,
            color=color,
            linewidth=linewidth
        )

        # Draw sampled points
        plt.scatter(
            x,
            y,
            c=color,
            s=point_size
        )

    plt.gca().invert_yaxis()
    plt.axis("off")
    plt.title("Interpolated Contours")
    plt.savefig(os.path.join(path, "interpolated_contours_{}.png".format(datetime.now().strftime("%Y%m%d_%H%M%S"))))
    #plt.show()
    #remove to see figure
    plt.close()
    
    
def find_contour_with_threshold(mask, arbitrary_minimal_area: int=1000, arbitrary_maximal_area: int=100000,plot: bool = True):
    '''     
    Find contours in a binary mask using OpenCV.
    Args:
        mask (numpy.ndarray): Binary mask where contours are to be found.
        arbitrary_minimal_area (int): Minimum area for a contour to be considered (default = 1000)
        plot (bool): Whether to display the contours (default = True)
    Returns:
        list: A list of contours, where each contour is an array of (x, y) coordinates.
    '''
    contours, hierarchy = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    contours = [cnt[:, 0, :] for cnt in contours]

    large_contours = []

    for c in contours:
        area = cv2.contourArea(c)
        if area > arbitrary_minimal_area and area < arbitrary_maximal_area:
            large_contours.append(c)
            
    if plot:
        #print(f"Number of large contours: {len(large_contours)}")
        result = mask.copy()
        gray_image = (result * 255).astype(np.uint8)
        result = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

        cv2.drawContours(result, large_contours,-1,(0, 0, 255),3)

        plt.figure(figsize=(10,8))
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        
        plt.savefig(os.path.join("..", "Project/Rapport", "contours_{}.png".format(datetime.now().strftime("%Y%m%d_%H%M%S"))))
        #plt.show()
        #remove to see figure
        plt.close()
    return large_contours


# FOR NOW I HAND SELECTED THE NBRE OF CONTOURS- USE THRESHOLD VALUE INSTEAD
# try to change the selction with areamin = 20k and aspect ratio > 23 to remove leaves (careful for player!!)
def relevant_contours_finder(mask,contours, contours_to_consider, infos_and_plot:bool=True, minimal_ar = 20, path=""):
    
    #calciulates aspect ratios for each contour
    aspect_ratios = [cv2.contourArea(cnt) / cv2.arcLength(cnt, True) if cv2.arcLength(cnt, True) > 0 else 0 for cnt in contours]
    
    # Find indices of n largest aspect ratios
    number_of_contours = contours_to_consider if contours_to_consider < len(contours) else len(contours)

    sorted_indices = np.argsort(aspect_ratios)[-number_of_contours:][::-1]

    #keep only contours above minimal aspect ratio
    filtered_indices = [i for i in sorted_indices if aspect_ratios[i] > minimal_ar]

    sorted_contours = [contours[i] for i in filtered_indices]

    if infos_and_plot:
        #print(f"Aspect ratios of the top {len(sorted_contours)} contours: {[aspect_ratios[i] for i in filtered_indices]}")
        #print("Areas of top aspect ratio contours:", [cv2.contourArea(contours[i]) for i in filtered_indices])
        
        result = mask.copy()
        
        gray_image = (result * 255).astype(np.uint8)
        
        result = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

        cv2.drawContours(result,sorted_contours,-1,(0, 0, 255),3)

        # add simple index labels (0,1,2,...)
        for idx, cnt in enumerate(sorted_contours):

            M = cv2.moments(cnt)

            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = cnt[0][0]

            cv2.putText(
                result,
                str(idx),
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        plt.figure(figsize=(10,8))
        
        plt.title(f"Top {len(sorted_contours)} contours with highest aspect ratio", fontsize=16)
        
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        

        plt.savefig(os.path.join(path, "top_contours_{}.png".format(datetime.now().strftime("%Y%m%d_%H%M%S"))))
        
        plt.axis("off")
        
        #plt.show()
        #remove to see figure
        plt.close()
        
    return sorted_contours

def merging_mask_calculator(contours, distances, min_distance_threshold=300, max_distance_threshold=350):
    '''
    arguments:
        - contours: list of contours to merge  
        - distances: distance matrix between contours
        - min_distance_threshold: minimum distance to consider for merging
        - max_distance_threshold: maximum distance to consider for merging
        returns:
        - merging_mask: binary matrix indicating which contours were merged
    '''
    
    N = contours.shape[0]
    merging_mask = np.zeros_like(distances)
    for i in range(N):
        for j in range(i + 1, N):
            if min_distance_threshold < distances[i, j] < max_distance_threshold:
                merging_mask[i, j] = 1
                merging_mask[j, i] = 1

    return merging_mask



def merge_contours_from_mask(contours, merging_mask):
    '''
    arguments:
        - contours: list of contours to merge  
        - merging_mask: binary matrix indicating which contours to merge

    returns:
        - merged_contours_untested: list of merged contours (not tested yet)
    '''
    

    N = contours.shape[0]
    merged_contours = copy.deepcopy(contours)
    merged_contours_untested = []

    for i in range(N):
        for j in range(i + 1, N):   # avoid duplicates and self-pairs
            if merging_mask[i, j] == 1:
                # merge contour i and contour j
                merged_contour = np.vstack((
                    merged_contours[i],
                    merged_contours[j]
                ))

                merged_contours_untested.append(merged_contour)
            
                
    return merged_contours_untested



def plot_bounding_boxes(mask,bounding_boxes,show_center=True,show_area=True,show_coordinates=True,figsize=(10, 8), color_tested="None", path =""):
    """
    Plot rotated bounding boxes with labels.

    arguments:
        - mask: binary mask or grayscale image
        - bounding_boxes: list of cv2.minAreaRect outputs
        - show_center: display center point
        - show_area: display rectangle area
        - show_coordinates: display center coordinates
        - figsize: matplotlib figure size
        - path : path to save the image
    """

    # prepare image
    result = mask.copy()

    if len(result.shape) == 2:
        gray_image = (result * 255).astype(np.uint8)
        result = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

    for bbox in bounding_boxes:

        (x, y), (w, h), angle = bbox

        # rotated rectangle
        rect = ((x, y), (w, h), angle)
        box = cv2.boxPoints(rect)
        box = box.astype(np.int32)
        cv2.drawContours(result, [box], 0, (0, 255, 0), 2)

        # center
        center_x = int(x)
        center_y = int(y)

        if show_center:
            cv2.circle(
                result,
                (center_x, center_y),
                10,
                (255, 0, 0),
                -1
            )

        # label text
        labels = []

        if show_coordinates:
            labels.append(f"({center_x},{center_y})")

        if show_area:
            area = int(w * h)
            labels.append(f"A={area}")
            
        labels.append(f"Tested: {color_tested}")

        label_text = " | ".join(labels)

        # draw label
        cv2.putText(
            result,
            label_text,
            (center_x + 10, center_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 0, 0),
            6,
            cv2.LINE_AA
        )

    # display
    plt.figure(figsize=figsize)
    plt.title("Original mask with bounding boxes")
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.savefig(os.path.join(path, "bounding_boxes_{}.png".format(datetime.now().strftime("%Y%m%d_%H%M%S"))))
    #plt.show()
    #remove to see figure
    plt.close()


def compute_distance_matrix(contours):
    N = contours.shape[0]
    #print(f"Number of contours: {N}")

    distances = np.zeros((N, N))
    for cnt in range(N):
        for cnt2 in range(N):
            if cnt != cnt2:
                distances[cnt, cnt2] = np.mean(np.linalg.norm(contours[cnt] - contours[cnt2], axis=1))
            else :
                distances[cnt, cnt2] = np.inf        
    return distances
    
    
def merging_verifyer(contours, merging_mask, w,d, margin=10):
    '''
    arguments:
        - contours: list of contours to merge  
        - merging_mask: binary matrix indicating which contours were merged
        - (w,d): width and depth that the bounding boxes should have to be considered valid
    returns:
        - verified_merged_contours: list of merged contours that are verified to be valid
    '''
    new_merging_mask = np.zeros_like(merging_mask)
    merged_contour = []
    for i in range(merging_mask.shape[0]):
        for j in range(i+1, merging_mask.shape[1]):
            if merging_mask[i, j]: #if these two contours were merged
               # merge contour i and contour j
                merged_contour = np.vstack((contours[i],contours[j]))
                # OpenCV contour format
                cnt_cv = merged_contour.astype(np.int32).reshape((-1, 1, 2))
                #check bounding box
                bbox = cv2.minAreaRect(cnt_cv)
                bw, bh = bbox[1]
                #print("Box for ", i, " and ", j, "is ", bw, bh)
                
                if ((abs(bw - w) <= margin and abs(bh - d) <= margin) or (abs(bw - d) <= margin and abs(bh - w) <= margin)):                    
                    new_merging_mask[i, j] = 1 #if valid, keep the merging
                    new_merging_mask[j, i] = 1 #symmetric
                    
    return new_merging_mask



def _is_rect_inside(rect_pts, outer_pts):
    """
    Check if rotated rectangle (rect_pts) is fully inside another polygon (outer_pts)
    """
    for p in rect_pts:
        # pointPolygonTest: >= 0 means inside or on edge
        if cv2.pointPolygonTest(outer_pts, (float(p[0]), float(p[1])), False) < 0:
            return False
    return True


def remove_nested_contours(contours):
    '''
    Remove contours that are fully nested inside another contour.

    Uses minAreaRect (oriented bounding boxes) for robustness.
    '''

    contours = [_clean_contour(c) for c in contours]
    # Precompute rotated rectangles for all contours
    rects = []
    rect_pts_list = []

    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        rects.append(rect)
        rect_pts_list.append(box)

    keep = [True] * len(contours)

    for i in range(len(contours)):
        for j in range(len(contours)):
            if i == j:
                continue

            # if i is inside j → remove i
            if _is_rect_inside(rect_pts_list[i], rect_pts_list[j]):
                keep[i] = False
                break

    filtered_contours = [
        cnt for cnt, k in zip(contours, keep) if k
    ]

    return filtered_contours


def _clean_contour(cnt):
    cnt = np.asarray(cnt)

    # ensure correct dtype
    cnt = cnt.astype(np.float32)

    # ensure contiguous memory (important for OpenCV)
    cnt = np.ascontiguousarray(cnt)

    return cnt