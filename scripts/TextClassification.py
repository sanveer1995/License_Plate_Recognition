from skimage.transform import resize
from sklearn.externals import joblib
import templatematching
import os


def classify_objects(objects, model, tuple_resize):
    letters = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
            'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z'
        ]
    classificationResult = []
    for eachObject in objects:
        eachObject = resize(eachObject, tuple_resize)
        eachObject = eachObject.reshape(1, -1)
        result = model.predict(eachObject)
        probabilities = model.predict_proba(eachObject)
        result_index = letters.index(result[0].decode('utf-8'))
        prediction_probability = probabilities[0, result_index]
        # template matching when necessary
        if result[0] in templatematching.confusing_chars and prediction_probability < 0.15:
            print('here')
            result[0] = templatematching.template_match(result[0],
                eachObject, os.path.join(current_dir, 'training_data', 'train20X20'))
        classificationResult.append(result)
        
    return classificationResult


def text_reconstruction(plate_string, position_list):
    posListCopy = position_list[:]
    position_list.sort()
    rightplate_string = ''
    for each in position_list:
        rightplate_string += plate_string[posListCopy.index(each)]
            
    return rightplate_string


def get_text(char_candidates):
	current_dir = os.getcwd()
	model_dir = os.path.join(current_dir, 'ml_models/SVC_model/SVC_model.pkl')
	model = joblib.load(model_dir)

	text_result = classify_objects(char_candidates['fullscale'], model, (20, 20))

	scattered_plate_text = ''
	for eachPredict in text_result:
		scattered_plate_text += eachPredict[0].decode('utf-8')

	reconstructed_plate_text = text_reconstruction(scattered_plate_text, char_candidates['columnsVal'])
	return reconstructed_plate_text

