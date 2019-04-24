from darkflow.net.build import TFNet

def load_convnet():
	options = {
  	"model": "cfg/tiny-yolo-voc-lp.cfg",
  	"load": -1,
  	'threshold': 0.3  
	}

	return TFNet(options)


def detect_plates(img):
	tfnet = load_convnet()
	results = tfnet.return_predict(img)
	return results
