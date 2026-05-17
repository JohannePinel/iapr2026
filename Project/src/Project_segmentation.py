from skimage.color import rgb2hsv
from skimage.morphology import closing, opening, disk, remove_small_holes, remove_small_objects, binary_dilation

from skimage.measure import label, regionprops


import numpy as np

from src.Project_plots import *

# Ce fichier contient les fonctions pour segmenter les cartes dans l'image, classer les cartes détectées en carte centrale et cartes des joueurs, et détecter le joueur actif basé sur la position d'un marker (petit objet). Il inclut des fonctions pour appliquer des seuils de couleur en HSV, traiter les masques pour trouver les régions de cartes, et fusionner les cartes proches pour corriger les erreurs de segmentation.


def extract_hsv_channels(img):

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


def apply_hsv_threshold(img, h_thresh=0.01, s_thresh=0.1, v_thresh=0.1):
    """
    Apply threshold to the input image in hsv colorspace.

    Args
    ----
    img: np.ndarray (M, N, C)
        Input image of shape MxN and C channels.
    h_thresh: float
        Hue value threshold
    s_thresh: float
        Saturation value threshold
    v_thresh: float
        Value threshold
    
    Return
    ------
    img_th: np.ndarray (M, N)
        Thresholded image.
    """

    # Define the default value for the input image
    M, N, C = np.shape(img)
    img_th = np.zeros((M, N))

    # Use the previous function to extract HSV channels
    data_h, data_s, data_v = extract_hsv_channels(img=img)
    
    img_th = (data_h > h_thresh) & (data_s > s_thresh) & (data_v > v_thresh)
    
    return  img_th


def apply_closing(img_th, disk_size):

    img_closing = np.zeros_like(img_th)
    img_closing = closing(img_th, disk(disk_size))

    return img_closing

def apply_opening(img_th, disk_size):

    img_opening = np.zeros_like(img_th)

    img_opening = opening(img_th, disk(disk_size))

    return img_opening

def remove_objects(img_th, size):

    img_obj = np.zeros_like(img_th)
    img_obj = remove_small_objects(img_th, min_size=size)

    return img_obj


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
        v_min, v_max = 0.9, 1.0

    if mode == "green":
        h_min, h_max = 0.22, 0.40
        s_min, s_max = 0.3, 0.7
        v_min, v_max = 0.6, 1.0

    if mode == "black":
        h_min, h_max = 0.0, 1.0  
        s_min, s_max = 0.0, 0.5 #was 0.6
        v_min, v_max = 0.0, 0.4  

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
    disk_size: int = 2,
    object_min_size: int = 30,
):


    img_th = apply_hsv_threshold_v2(img, mode=mode)

    out_mask = apply_closing(img_th, disk_size)
    out_mask = apply_opening(out_mask, disk_size)
    out_mask = remove_small_objects(out_mask, min_size=object_min_size)


    return out_mask

############# detection des cartes #############

def detect_cards_from_masks(masks_dict, min_area=15000, max_area=50000):
    'Trouve les cartes à partir des masques de couleur et retourne une liste de cartes détectées avec leurs propriétés (couleur, bbox, centroid, area, orientation)'
    'Returne aussi les petits objets détectés pour aider à la détection du joueur actif'
    
    all_cards = []
    small_objects = []

    
    for color_name, mask in masks_dict.items():

        labeled_mask = label(mask.astype(bool))
        regions = regionprops(labeled_mask)
        
        for region in regions:
            if min_area < region.area < max_area:
                minr, minc, maxr, maxc = region.bbox
                
                card_info = {
                    'color': color_name,
                    'bbox': (minr, minc, maxr, maxc),
                    'centroid': region.centroid,
                    'area': region.area,
                    'orientation': region.orientation,
                }
                
                all_cards.append(card_info)
            
            elif 100 < region.area < 5000:  
                small_objects.append({
                    'color': color_name,
                    'bbox': region.bbox,
                    'centroid': region.centroid,
                    'area': region.area
                })
                #print(f"Small object detected: color={color_name}, area={region.area}, centroid={region.centroid}")


    return all_cards, small_objects


def classify_by_position(cards, image_shape, small_objects):
    'Selon leur position dans l\'image, classifie les cartes en carte centrale ou cartes des joueurs. Retourne aussi le joueur actif basé sur la position du marker (petit objet)'

    h, w = image_shape[:2]
    center_y, center_x = h / 2, w / 2

    center_radius = min(h, w) * 0.15

    center_candidates = []
    player_cards_list = []

    for card in cards:
        cy, cx = card['centroid']
        dist_to_center = np.sqrt((cy - center_y)**2 + (cx - center_x)**2)

        if dist_to_center < center_radius:
            center_candidates.append((dist_to_center, card))
        else:
            player_cards_list.append(card)

    center_card = None
    if center_candidates:
        center_candidates.sort(key=lambda x: x[0])
        center_card = center_candidates[0][1]
        for _, card in center_candidates[1:]:
            player_cards_list.append(card)

    player_cards = group_cards_by_player(player_cards_list, image_shape)

    active_player = detect_active_player(small_objects, image_shape)

    return center_card, player_cards, active_player


def detect_active_player(small_objects, image_shape):
    """
    Active player = position du marker dans l'image
    """

    if not small_objects:
        return "unknown"

    marker_y, marker_x = small_objects[-1]["centroid"]

    h, w = image_shape[:2]
    center_y, center_x = h / 2, w / 2

    dy = marker_y - center_y
    dx = marker_x - center_x

    # décision directionnelle (ROBUSTE)
    if abs(dy) > abs(dx):
        if dy > 0:
            return "p1"   # bottom
        else:
            return "p3"   # top
    else:
        if dx > 0:
            return "p2"   # right
        else:
            return "p4"   # left
        

def merge_close_cards(cards, distance_threshold=500):
    'Pour les cartes détectées qui sont très proches les unes des autres, fusionne-les en une seule carte avec une bbox englobante et un centroid moyen. Cela aide à corriger les erreurs de segmentation qui peuvent découper une carte en plusieurs morceaux.'
    merged = []
    used = [False] * len(cards)

    for i in range(len(cards)):
        if used[i]:
            continue

        group = [cards[i]]
        used[i] = True

        cy1, cx1 = cards[i]['centroid']

        for j in range(i + 1, len(cards)):
            if used[j]:
                continue

            cy2, cx2 = cards[j]['centroid']

            dist = np.sqrt((cy1 - cy2)**2 + (cx1 - cx2)**2)

            if dist < distance_threshold:
                group.append(cards[j])
                used[j] = True

        if len(group) == 1:
            merged.append(group[0])
        else:
            mean_cy = np.mean([c['centroid'][0] for c in group])
            mean_cx = np.mean([c['centroid'][1] for c in group])

            minr = min(c['bbox'][0] for c in group)
            minc = min(c['bbox'][1] for c in group)
            maxr = max(c['bbox'][2] for c in group)
            maxc = max(c['bbox'][3] for c in group)

            merged_card = {
                **group[0],
                'centroid': (mean_cy, mean_cx),
                'bbox': (minr, minc, maxr, maxc),
                'area': sum(c['area'] for c in group)
                
            }

            merged.append(merged_card)

    return merged


def group_cards_by_player(cards, image_shape, center_ratio=0.25):
    """
    center_ratio: fraction of image size considered as "center zone"
    """

    if len(cards) == 0:
        return {}

    h, w = image_shape[:2]
    center_y, center_x = h / 2, w / 2

    # size of the "no-player zone"
    center_h = h * center_ratio
    center_w = w * center_ratio

    players = {
        "player_1_bottom": [],
        "player_2_right": [],
        "player_3_top": [],
        "player_4_left": [],
    }

    for card in cards:
        cy, cx = card["centroid"]

        dy = cy - center_y
        dx = cx - center_x

        if abs(dy) < center_h / 2 and abs(dx) < center_w / 2:
            pass

        if abs(dy) > abs(dx):
            if dy > 0:
                players["player_1_bottom"].append(card)
            else:
                players["player_3_top"].append(card)
        else:
            if dx > 0:
                players["player_2_right"].append(card)
            else:
                players["player_4_left"].append(card)

    return players

'''
def main_detection(image, masks_dict):

  
    #print(f"Nombre total de cartes détectées: {len(all_cards)}")
    
    #print(f"Nombre de joueurs: {len([cards for cards in player_cards.values() if cards])}")

    #print(f"\nCarte centrale: {center_card['color'] if center_card else 'Non détectée'}")

   
    all_cards, small_objects = detect_cards_from_masks(masks_dict)

    all_cards = merge_close_cards(all_cards, distance_threshold=400)

    center_card, player_cards, active_player = classify_by_position(
        all_cards,
        image.shape,
        small_objects   
    )

    for player_name, cards in player_cards.items():
        #print(f"{player_name}: {len(cards)} cartes - Couleurs: {[c['color'] for c in cards]}")
        pass
        
    
    return center_card, player_cards, all_cards, active_player
'''

