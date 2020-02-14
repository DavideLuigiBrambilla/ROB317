import numpy as np
import matplotlib.pyplot as plt
import cv2
from numpy import shape

##Reads the video from file
capture = cv2.VideoCapture("TP2_Videos/Extrait1-Cosmos_Laundromat1(340p).m4v")

## Reads video from webcam
# ~ capture = cv2.VideoCapture(0)

count_hist = 0

def Calcule_2D_YUV_histogramme(capture):
	bins = 64
	(grabbed, frame) = capture.read()
	if grabbed:
		# Normalize histograms based on number of pixels per frame.
		numPixels = np.prod(frame.shape[:2])
		
		#Read image & transform in YUV
		yuv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

		#Color_map for the YUV
		color_uv = np.zeros((bins, bins, 3), np.uint8)
		u, v = np.indices(color_uv.shape[:2])
		color_uv[:,:,0] = 50
		color_uv[:,:,1] = u * 256/bins
		color_uv[:,:,2] = v * 256/bins
		color_uv = cv2.cvtColor(color_uv, cv2.COLOR_YUV2BGR)

		### 2D Histogram
		h = cv2.calcHist([yuv_image], [1, 2], None, [bins]*2, [-0, 256]*2)
		
		## Normalisation non-lineaire/ lineaire
		h[:,:] = np.log(h[:,:])
		hist_norm = np.clip(h, 0, np.max(h))
		hist_norm[:,:] = (hist_norm[:,:]/np.max(hist_norm))
		
		## Aplication de la mask color_uv
		hist_mask = color_uv*hist_norm[:,:,np.newaxis] / bins
		
		### Display the images
		## Video
		cv2.namedWindow('Video converted to Yuv')     
		cv2.moveWindow('Video converted to Yuv', 500,0)  
		cv2.imshow('Video converted to Yuv', frame)
		
		## Histogramme avec la masque
		cv2.namedWindow('2D Histogram (u,v) components', cv2.WINDOW_NORMAL)
		cv2.resizeWindow('2D Histogram (u,v) components', 500,500)
		cv2.imshow('2D Histogram (u,v) components', hist_mask)	
		
		## Histogramme avec la probabilité jointe
		plt.figure(num=2)
		plt.clf()
		plt.title("Hisogramme 2D des composantes u et v")
		plt.xlabel("Composante v")
		plt.ylabel("Composante u")
		plt.imshow(hist_norm,interpolation = 'nearest')
		plt.colorbar()
		plt.draw();
	
		return hist_mask
		
def plot_correlation(corr):
	plt.figure(num=1, figsize=(4, 4))
	plt.clf()
	plt.rcParams["figure.figsize"] = (5,5)
	plt.plot(corr, 'b', linewidth = 0.5)
	plt.ylim([0, 1])
	plt.title("Correlation des histogrammes h et h-1")
	plt.xlabel("Numero de frames")
	plt.ylabel("Correlation (%)")
	plt.draw()
	plt.pause(0.0001)	

## Initialise les variables pour la comparations des histogrammes	
hist_mask = np.zeros_like(Calcule_2D_YUV_histogramme(capture))
hist_mask_old = np.zeros_like(Calcule_2D_YUV_histogramme(capture))
corr = []

while True:
	(grabbed, frame) = capture.read()
	if not grabbed:
		break
	
	## Sauvegard l'ancien et le nouveau histogramme
	hist_mask_old = hist_mask
	hist_mask = Calcule_2D_YUV_histogramme(capture)
	
	## Correlation des deux histogrammes pour verifier le changement de plan
	corr.append(cv2.compareHist(hist_mask_old, hist_mask, cv2.HISTCMP_CORREL))
	
	## Plot en temps réel de la valeur de la correlation
	plot_correlation(corr)
	
	count_hist+=1
		
	# Type "q" to quit
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

capture.release()
cv2.destroyAllWindows()
