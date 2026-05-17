import os
from pathlib import Path
import string
import matplotlib.image as mpimg
import pandas as pd
from src.Project_segmentation import *
from src.Project_csv import *
from src.Project_cards_detection import *
from tqdm import tqdm
from src.Project_segmentation import *

# Ce fichier contient les fonctions pour formater les résultats de la détection en un format CSV conforme aux exigences de la compétition. Il inclut des fonctions pour formater les cartes détectées, créer les lignes du CSV, et traiter un dossier d'images pour générer le CSV final de soumission.

def format_card_for_csv(card):
    """
    Formate une carte selon le format requis: couleur_valeur
    Ex: r_5, b_skip, y_draw_2, wild
    """
    
    if card is None:
        return "bloublou"  # Carte par défaut si non détectée
    
    color = card.get('color', '').lower()
    
    # Donc on met un placeholder '0' en attendant la détection des chiffres
    value = card.get('value', '0')  # Par défaut '0' si pas de valeur A changer quand nicolas a trouver comment faire
    
    # Gérer les cartes spéciales
    if color == 'wild' or color == 'black':
        if 'draw_4' in str(value).lower() or 'draw4' in str(value).lower():
            return "wild_draw_4"
        return "wild"
    
    # Mapping des couleurs
    color_map = {
        'red': 'r',
        'blue': 'b',
        'green': 'g',
        'yellow': 'y',
        'black': 'wild'  # Carte noire = wild
    }
    
    color_code = color_map.get(color, 'r')  # Par défaut 'r' si couleur inconnue
    
    return f"{color_code}_{value}"


def format_player_cards(cards_list):
    """
    Formate la liste de cartes d'un joueur
    Retourne "EMPTY" si pas de cartes, sinon "card1;card2;card3"
    """
    if not cards_list or len(cards_list) == 0:
        return "EMPTY"
    
    
    formatted_cards = [format_card_for_csv(card) for card in cards_list]
    return ";".join(formatted_cards)




def create_submission_row(image_id, center_card, player_cards, active_player):
    """
    Crée une ligne du CSV de soumission
    """

    center_card_str = format_card_for_csv(center_card)

    player_order = [
        'player_1_bottom',
        'player_2_right',
        'player_3_top',
        'player_4_left'
    ]

    player_cards_formatted = []

    for player in player_order:
        cards = player_cards.get(player, [])
        player_cards_formatted.append(format_player_cards(cards))

    return {
        'image_id': image_id,
        'center_card': center_card_str,
        'active_player': active_player,
        'player_1_cards': player_cards_formatted[0],
        'player_2_cards': player_cards_formatted[1],
        'player_3_cards': player_cards_formatted[2],
        'player_4_cards': player_cards_formatted[3]
    }

def process_single_image(img_path, visualize=False):
    """
    Traite une seule image et retourne les résultats
    """
    # Charger l'image
    img = mpimg.imread(str(img_path))
    
    # Détecter les masques de couleur
    color_masks = {
        'red': find_area_by_type(img, mode='red'),
        'blue': find_area_by_type(img, mode='blue'),
        'yellow': find_area_by_type(img, mode='yellow'),
        'green': find_area_by_type(img, mode='green'),
        'black': find_area_by_type(img, mode='black'),
    }
    
    center_card, player_cards, all_cards, active_player = main_detection(img, color_masks)
        
    return center_card, player_cards, all_cards, active_player


def test_on_folder(folder_path, output_csv='test_results.csv', max_images=None, visualize=False):
    """
    Teste le pipeline sur plusieurs images d'un dossier
    """
    print(os.getcwd())
    
    test_path = Path(folder_path)
    image_extensions = ['*.jpg']
    test_images = []
    for ext in image_extensions:
        test_images.extend(test_path.glob(ext))
    
    test_images = sorted(test_images)
    
    # Limiter le nombre d'images si spécifié
    if max_images is not None:
        test_images = test_images[:max_images]
    
    print(f"\nDossier: {folder_path}")
    print(f"Nombre d'images à traiter: {len(test_images)}")
    
    if len(test_images) == 0:
        print(" Aucune image trouvée!")
        return None
    
    results = []
    
    print("\n Début du traitement...\n")
    
    for img_path in tqdm(test_images, desc="Traitement des images"):
        image_id = img_path.stem
        
        
        img = mpimg.imread(str(img_path))
        print("Image loaded:", img_path.name)
        
        # Détecter les masques de couleur
        color_masks = {
            'red': find_area_by_type(img, mode='red'),
            'blue': find_area_by_type(img, mode='blue'),
            'yellow': find_area_by_type(img, mode='yellow'),
            'green': find_area_by_type(img, mode='green'),
            'black': find_area_by_type(img, mode='black'),
        }
        
        # Détection principale
        center_card, player_cards, all_cards, active_player = main_detection(img, color_masks)
        
        # Créer la ligne de soumission
        row = create_submission_row(
            image_id,
            center_card,
            player_cards,
            active_player
        )        
        results.append(row)
        
        df = pd.DataFrame(results)
    
    df = df[['image_id', 'center_card', 'active_player', 
             'player_1_cards', 'player_2_cards', 'player_3_cards', 'player_4_cards']]
    
    # Sauvegarder le CSV
    df.to_csv(output_csv, index=False)
    
    
    print(f"\nTest terminé!")
    
    return df


def test_on_folder_contours(folder_path, output_csv='test_results.csv', output_folder= 'new_data' , max_images=None, visualize=False):
    """
    Teste le pipeline sur plusieurs images d'un dossier
    """
    print(os.getcwd())
    
    test_path = Path(folder_path)
    output_path = Path(output_folder)
    image_extensions = ['*.jpg']
    test_images = []
    for ext in image_extensions:
        test_images.extend(test_path.glob(ext))
    
    test_images = sorted(test_images)

    print("Resolved path:", test_path.resolve())
    print("output path :", output_path.resolve())
    
    # Limiter le nombre d'images si spécifié
    if max_images is not None:
        test_images = test_images[:max_images]

    print(f"\nDossier: {folder_path}")
    print(f"Nombre d'images à traiter: {len(test_images)}")
    
    if len(test_images) == 0:
        print(" Aucune image trouvée!")
        return None
    else: 
        print(f" Images trouvées: {[img.name for img in test_images]}")
    
    results = []
    
    print("\n Début du traitement...\n")
    
    for img_path in tqdm(test_images, desc="Traitement des images"):
        image_id = img_path.stem
        
        # create subfolder per image
        img_output_path = output_path / image_id
        img_output_path.mkdir(parents=True, exist_ok=True)
        
        img = mpimg.imread(str(img_path))
        print("Image loaded:", img_path.name)
        plt.imsave(img_output_path / img_path.name,img)
        
        # Détecter les masques de couleur
        color_masks = {
            'red': find_area_by_type(img, mode='red'),
            'blue': find_area_by_type(img, mode='blue'),
            'yellow': find_area_by_type(img, mode='yellow'),
            'green': find_area_by_type(img, mode='green'),
            'black': find_area_by_type(img, mode='black'),
        }
        print("Color masks computed.")
        all_cards = []
        for color in color_masks:
            print(" Color:", color)
            mask = color_masks[color]
            

            plot_thresholded_image(img=img,func=lambda img: mask,title=f"Combined detection in HSV space")
            contours = find_contour_with_threshold(mask, arbitrary_minimal_area=1000, arbitrary_maximal_area =50000, plot=True)
            contours_high_ar = relevant_contours_finder(mask, contours, contours_to_consider=20, infos_and_plot=True, minimal_ar = 10, path=img_output_path)

            new_contours = [c.reshape(-1, 2) for c in contours_high_ar]
            contours_inter = linear_interpolation(new_contours, n_samples=25) #now a list of all contoues with N points each
            plot_interpolated_contours(mask,contours_inter, path=img_output_path)
            
            distances = compute_distance_matrix(contours_inter)
            merging_mask = merging_mask_calculator(contours_inter, distances, min_distance_threshold=250, max_distance_threshold=500)
            #debug : pretty print mask
            #print("Merging mask:")
            #print(merging_mask)
            tested_merging_mask = merging_verifyer(contours_inter, merging_mask, w=290, d=480, margin=40)
            #print("Tested merging mask:")
            #print(tested_merging_mask)
            
            contours_all = contours_inter
            contours_merged = merge_contours_from_mask(contours_all, tested_merging_mask)
            unmerged_contours = [contours_inter[i] for i in range(len(contours_inter)) if not tested_merging_mask[i].any()]
            contours_all = contours_merged + unmerged_contours
            #remove contours which bboxes are fully inside other contour
            contours_all = [c.reshape(-1, 2) for c in contours_all]
            #print (f"Number of contours before removing nested: {len(contours_all)}")
            contours_all = remove_nested_contours(contours_all)
            print (f"Total number of contours: {len(contours_all)}")

            bounding_boxes = []

            #get infos for each contour
            for contour in contours_all:
                cnt_cv = contour.astype(np.int32).reshape((-1, 1, 2)) #openCV contour format
                bbox = cv2.minAreaRect(cnt_cv)
                bounding_boxes.append(bbox)
                card_info = {
                    'contour': contour,
                    'color': color,
                    'bbox': bbox,
                    'centroid': (bbox[0][1], bbox[0][0]),
                    'area': bbox[1][0] * bbox[1][1],
                    'orientation': bbox[2],
                }
                all_cards.append(card_info)
                
            plot_bounding_boxes(mask=mask,bounding_boxes=bounding_boxes, color_tested=color, show_center=True, show_area=False, show_coordinates=True, figsize=(10, 8), path=img_output_path)
        
       
        for i, card in enumerate(all_cards):
            color = card['color']
            position = card['centroid']
            angle = card['orientation']
            area = card['area']
            
            print(f"Card {i + 1}:")
            print(f"  Color:    {color}")
            print(f"  Position: {position}")
            print(f"  Angle:    {angle}")
            print(f"  Area:    {area} ")
            print("-" * 20)
            
        small_objects = []
        center_card, player_cards, active_player = classify_by_position(all_cards, img.shape, small_objects) 
        
        # Créer la ligne de soumission
        row = create_submission_row(
            image_id,
            center_card,
            player_cards,
            active_player
        )        
        results.append(row)
        
        df = pd.DataFrame(results)
    
    df = df[['image_id', 'center_card', 'active_player', 
             'player_1_cards', 'player_2_cards', 'player_3_cards', 'player_4_cards']]
    
    # Sauvegarder le CSV
    df.to_csv(output_csv, index=False)
    
    
    print(f"\nTest terminé!")
    
    return all_cards