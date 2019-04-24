import warnings
warnings.simplefilter("ignore", UserWarning)
import cv2
import os
from DetectPlate import detect_plates
from DetectChar import detect_chars
from TextClassification import get_text 


if __name__=='__main__':

	os.chdir('..')
	img_name = '3.png'

	img = cv2.imread("./LicPlateImages/"+img_name, cv2.IMREAD_COLOR)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	plates = detect_plates(img)
	
	if(len(plates)>0):

		for each_plate in plates:
			tl = (each_plate['topleft']['x'], each_plate['topleft']['y'])
			br = (each_plate['bottomright']['x'], each_plate['bottomright']['y'])

			img_label = img.copy()
			
			cv2.rectangle(img_label, tl, br, (255,0,0), 7)
			cv2.imwrite('./img_labels/'+img_name, cv2.cvtColor(img_label, cv2.COLOR_BGR2RGB))
			
			img_plate = img[tl[1]:br[1], tl[0]:br[0]]
			cv2.imwrite('./img_plates/'+img_name, cv2.cvtColor(img_plate, cv2.COLOR_BGR2RGB))

			confidence = '{:.5f}%'.format(each_plate['confidence']*100)
			print('License Plate Confidence :- '+confidence)

			chars = detect_chars('./img_plates/'+img_name)

			if(not chars == {}):
				plate_text = get_text(chars)
				print(plate_text)



