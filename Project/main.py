from src.arno_mainutils import test_on_folder_contours
from src.Project_plots import *
from src.Project_segmentation import *
from src.Project_csv import *
import matplotlib.image as mpimg
import cv2


if __name__ == "__main__":
    print(os.getcwd())
    folder_path = r"C:\Users\arnod\Desktop\AA EPFL MASTER\AA EPFL MA4\EE451 Image processing\PROJECT\iapr2026\Project\data\iapr-26-uno-vision-challenge\arno_images"
    ouput_folder = f"Results\\test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_path_csv = f"{ouput_folder}/submission_contours_only.csv"
    os.makedirs(ouput_folder, exist_ok=True)
    '''
    df = test_on_folder(
        folder_path="data/iapr-26-uno-vision-challenge/test_images_johanne",
        output_csv="submission_colors_only.csv",
        max_images=10  # Teste sur 10 images d'abord
    )
    '''
    
    all_cards = test_on_folder_contours(
        folder_path=folder_path,
        output_csv=output_path_csv,
        output_folder=ouput_folder,
        max_images=1  # Teste sur 1 images d'abord
    )
    
    
    