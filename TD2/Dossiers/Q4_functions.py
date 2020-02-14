import numpy as np
import cv2
from skimage.feature import hog
from numpy import shape

def Calcule_2D_YUV_histogramme(ret, frame):
	bins = 64
	if ret:
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
		hist_norm= cv2.calcHist([yuv_image], [1, 2], None, [bins]*2, [-0, 256]*2)
		
		## Normalisation non-lineaire/ lineaire
		hist_norm[:,:] = np.log(hist_norm[:,:])
		hist_norm = np.clip(hist_norm, 0, np.max(hist_norm))
		hist_norm[:,:] = (hist_norm[:,:]/np.max(hist_norm))
		
		return hist_norm

def show_video(frame):
	### Display the images
	cv2.imshow('Video originale', frame)
	
def Find_flow(prvs,next):
	flow = cv2.calcOpticalFlowFarneback(prvs,next,None, 
									pyr_scale = 0.5,# Taux de réduction pyramidal
									levels = 3, # Nombre de niveaux de la pyramide
									winsize = 15, # Taille de fenÃªtre de lissage (moyenne) des coefficients polynomiaux
									iterations = 3, # Nb d'itération par niveau
									poly_n = 7, # Taille voisinage pour approximation polynomiale
									poly_sigma = 1.5, # E-T Gaussienne pour calcul dérivés 
									flags = 0)	
	return flow

def Calcule_Flow_histogramme(flow):
	bins = 64
	hist = cv2.calcHist([flow], [0, 1], None, [bins]*2, [-bins, bins]*2)	
	
	## Elimination des valeurs statics (pas importantes dans ce cas)
	hist[hist[:,:]>np.std(hist)/2] = np.mean(hist)
	
	## Normalisation non-lineaire
	hist[:,:] = (hist[:,:])**0.5
	hist[:,:] = (hist[:,:]/np.max(hist))
	return hist
	
def apply_hog(image, number_ori):
	fd, hog_image = hog(image, orientations=number_ori, pixels_per_cell=(16, 16),
						cells_per_block=(1, 1), visualize=True)#, multichannel=True)
	number_histograms = (int(shape(image)[0]/16)*int(shape(image)[1]/16))
	teste = np.reshape(np.array(fd), (number_histograms, number_ori))
	histogram_hog = np.sum(teste, axis=0)
	histogram_hog = np.reshape(histogram_hog, (number_ori))
	histogram_hog[:] = (histogram_hog[:]/np.max(histogram_hog))*256
	return histogram_hog

def eqm_images(image1, image2):
	# 'Erreur quadratique moyenne' entre les images
	erreur = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
	erreur /= float(image1.shape[0] * image1.shape[1])
	return erreur

def compare_images(image1, image2):
	m = eqm_images(image1, image2)
	#s: Compute the mean structural similarity index between two images
	s = measure.compare_ssim(image1, image2, multichannel=True)
	return m*s
