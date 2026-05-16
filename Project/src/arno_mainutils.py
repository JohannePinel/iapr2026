import os
from pathlib import Path
import string
import matplotlib.image as mpimg
import pandas as pd
from src.Project_segmentation import *
from src.Project_csv import *
from src.arno_utils import *
from tqdm import tqdm
from src.Project_segmentation import find_area_by_type

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
            contours = find_contour_with_threshold(mask, arbitrary_minimal_area=1000, arbitrary_maximal_area =35000, plot=True)
            contours_high_ar = relevant_contours_finder(mask, contours, contours_to_consider=20, infos_and_plot=True, minimal_ar = 18, path=img_output_path)
            for cnt in contours_high_ar:
                print("Area = ", cv2.contourArea(cnt))
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
            print (f"Number of contours before removing nested: {len(contours_all)}")
            for cnt in contours_all:
                print(" Contour points:", cnt.shape)
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
                    'centroid': (bbox[0][0], bbox[0][1]),
                    'area': bbox[1][0] * bbox[1][1],
                    'orientation': bbox[2],
                }
                all_cards.append(card_info)
                
            plot_bounding_boxes(mask=mask,bounding_boxes=bounding_boxes, color_tested=color, show_center=True, show_area=True, show_coordinates=True, figsize=(10, 8), path=img_output_path)
        
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
    
    return df