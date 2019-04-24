import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from skimage.filters import threshold_otsu
from skimage.io import imread
from skimage.transform import resize
from skimage import measure
from skimage.measure import regionprops


def inverted_threshold(grayscale_image):
    threshold_value = threshold_otsu(grayscale_image) - 0.05
    return grayscale_image < threshold_value


def identify_boundary_objects(a_license_plate):
    labelImage = measure.label(a_license_plate)
    #character_dimensions = (0.2*a_license_plate.shape[0], 0.90*a_license_plate.shape[0], 0.03*a_license_plate.shape[1], 0.15*a_license_plate.shape[1])
    #minHeight, maxHeight, minWidth, maxWidth = character_dimensions
    regionLists = regionprops(labelImage)
    return regionLists


def detect_chars(img_path):
	license_plate_grayscale = imread(img_path, as_gray=True)
	license_plate = inverted_threshold(license_plate_grayscale)

	character_objects = identify_boundary_objects(license_plate)

	cord = []
	counter=0
	column_list = []

	fig, ax1 = plt.subplots(1)
	ax1.imshow(license_plate, cmap="gray")

	character_dimensions = (0.2*license_plate.shape[0], 0.90*license_plate.shape[0], 0.03*license_plate.shape[1], 0.15*license_plate.shape[1])
	minHeight, maxHeight, minWidth, maxWidth = character_dimensions
	for regions in character_objects:
	    minimumRow, minimumCol, maximumRow, maximumCol = regions.bbox
	    character_height = maximumRow - minimumRow
	    character_width = maximumCol - minimumCol
	    roi = license_plate[minimumRow:maximumRow, minimumCol:maximumCol]
	    if character_height > minHeight and character_height < maxHeight and character_width > minWidth and character_width < maxWidth:
	        if counter == 0:
	            samples = resize(roi, (20,20))
	            cord.append(regions.bbox)
	            counter += 1
	        elif counter == 1:
	            roismall = resize(roi, (20,20))
	            samples = np.concatenate((samples[None,:,:], roismall[None,:,:]), axis=0)
	            cord.append(regions.bbox)
	            counter+=1
	        else:
	            roismall = resize(roi, (20,20))
	            samples = np.concatenate((samples[:,:,:], roismall[None,:,:]), axis=0)
	            cord.append(regions.bbox)
	        column_list.append(minimumCol)
	        rect_border = patches.Rectangle((minimumCol, minimumRow), maximumCol - minimumCol, maximumRow - minimumRow, edgecolor="red",
	                                       linewidth=2, fill=False)
	        ax1.add_patch(rect_border)
	

	if len(column_list) == 0:
	    char_candidates = {}
	    print("No character was segmented")
	else:
	    char_candidates = {
	                'fullscale': samples,
	                'coordinates': np.array(cord),
	                'columnsVal': column_list
	                }
	    #plt.show()

	return char_candidates
