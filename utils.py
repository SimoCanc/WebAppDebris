import io
import sys
import numpy as np
import matplolib.pyplot as plt
from scipy.ndimage import label

def find_highest_lowest_xy(mask):
    '''Estrapolazione coordinate estremi maschera''' 
    # Estrapolazione regioni indipendenti formate da pixel 0
    labeled_mask, num_features = label(mask == 0)
    
    if num_features == 0: 
        sys.exit('Nessuna Maschera Rilevata')
    
    # Conteggio pixel per regione
    sizes = [np.sum(labeled_mask == i) for i in range(1, num_features + 1)]
    
    # Calcolo regione avente più pixel
    largest_index = np.argmax(sizes) + 1  
    
    # Calcolo coordinate regione più grande
    largest_region_coords = np.argwhere(labeled_mask == largest_index)
    
    # Calcolo coordinate degli estremi della regione più grande
    ymin = np.min(largest_region_coords[:, 0])
    ymax = np.max(largest_region_coords[:, 0])
    xmin = np.min(largest_region_coords[:, 1])
    xmax = np.max(largest_region_coords[:, 1])
    
    if ymin == ymax:  # In caso di regione orizzontale
        y_near_mask = [ymin, ymin]
        x_near_mask = [xmin, xmax]
    elif xmin == xmax:  # In caso di regione verticale
        y_near_mask = [ymin, ymax]
        x_near_mask = [xmin, xmin]
    else:  # Per le altre forme
        dist_ymin = abs(largest_region_coords[:, 0] - ymin)
        dist_ymax = abs(largest_region_coords[:, 0] - ymax)
        dist_xmin = abs(largest_region_coords[:, 1] - xmin)
        dist_xmax = abs(largest_region_coords[:, 1] - xmax)
        y_near_mask = [ymin, ymax]
        x_near_mask = [largest_region_coords[np.argmin(dist_ymin), 1], largest_region_coords[np.argmin(dist_ymax), 1]]
    
    return (y_near_mask, x_near_mask)

#################################################################################################################################
def from_16_to_8(file_img, img_size):
    '''Ridimensionamento e trasformazione a 8-bit dell'immagine'''
    image = PILImage.create(file_img)
    if image.size != (img_size, img_size):
        image = image.resize((img_size, img_size))
    image_8bit = image.point(lambda x: x * (1.0 / 256)).convert('L')
    return image_8bit

#################################################################################################################################	
def plot_coords(img_8bit):
    '''Graficazione coordinate'''
    plt.figure(figsize=(12, 12))
    image = plt.imread(img_8bit)
    plt.imshow(image, cmap='gray') 
    plt.scatter(x_coords, y_coords, color='orange') 
    plt.show()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf
	