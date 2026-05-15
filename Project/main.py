from src.Project_plots import *
from src.Project_segmentation import *
from src.Project_csv import *
import matplotlib.image as mpimg
import cv2


if __name__ == "__main__":
    print(os.getcwd())

    df = test_on_folder(
        folder_path="data/iapr-26-uno-vision-challenge/test_images_johanne", 
        output_csv="submission_colors_only.csv",
        max_images=10  # Teste sur 10 images d'abord

    )
    