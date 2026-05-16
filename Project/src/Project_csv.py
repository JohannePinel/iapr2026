import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.image as mpimg

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