import cv2
import numpy as np
import pickle

def grid(img):
	x, y, w, h, = 420, 140, 10, 10
	d = 10
	imgCrop = None
	crop = None
	for i in range(10):
		for j in range(5):
			if np.any(imgCrop == None):
				imgCrop = img[y : y + h, x : x + w]
			else:
				imgCrop = np.hstack((imgCrop, img[y : y + h, x : x + w]))
			cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 1)
			x += (w + d)
		if np.any(crop == None):
			crop = imgCrop
		else:
			crop = np.vstack((crop, imgCrop))
		imgCrop = None
		x = 420
		y += (h + d)
	return crop

def lighting():
	cam = cv2.VideoCapture(1)
	if cam.read()[0] == False:
		cam = cv2.VideoCapture(0)
	x, y, w, h = 300, 100, 300, 300
	keyC, keyS = False, False
	imgCrop = None
	while True:
		img = cam.read()[1]
		img = cv2.flip(img, 1)
		img = cv2.resize(img, (640, 480))
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

		keyPressed = cv2.waitKey(1)
		if keyPressed == ord('c'):
			hsvCrop = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2HSV)
			keyC = True
			hist = cv2.calcHist([hsvCrop], [0, 1], None, [180, 256], [0, 180, 0, 256])
			cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

		elif keyPressed == ord('s'):
			keyS = True
			break

		if keyC:
			dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
			dstCopy = dst.copy()
			disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
			cv2.filter2D(dst, -1, disc, dst)
			blur = cv2.GaussianBlur(dst, (11, 11), 0)
			blur = cv2.medianBlur(blur, 15)
			__, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
			thresh = cv2.merge((thresh, thresh, thresh))
			cv2.imshow("thresh", thresh)
		if not keyS:
			imgCrop = grid(img)
		cv2.imshow("lighting", img)


	cam.release()
	cv2.destroyAllWindows()

	with open("grid", "wb") as f:
		pickle.dump(hist, f)

lighting()