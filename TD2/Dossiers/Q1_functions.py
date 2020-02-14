import cv2
from skimage.feature import hog
from numpy import shape
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt

def apply_hog(image, number_ori):
	fd, hog_image = hog(image, orientations=number_ori, pixels_per_cell=(16, 16),
						cells_per_block=(1, 1), visualize=True)#, multichannel=True)
	number_histograms = (int(shape(image)[0]/16)*int(shape(image)[1]/16))
	teste = np.reshape(np.array(fd), (number_histograms, number_ori))
	histogram_hog = np.sum(teste, axis=0)
	histogram_hog = np.reshape(histogram_hog, (number_ori))
	histogram_hog[:] = (histogram_hog[:]/np.max(histogram_hog))*256
	
	
	
	# Plot le histogramme noir et blanc
	plt.figure(num=11, figsize=(4, 4))
	plt.clf()
	plt.title('HOG Histogramme avec %i orientations' %number_ori)
	plt.xlabel("Orientation axis")
	plt.ylabel("Fréquence de détéction de la direction")
	bars = np.zeros(number_ori)
	y_pos = np.arange(len(bars))
	plt.bar(y_pos,np.array(histogram_hog))
	plt.draw()
	plt.pause(1e-3)
	return histogram_hog

def Calcule_BW_histogramme(capture):	
	(grabbed, frame) = capture.read()
	bw_image = frame
	gray_hist = cv2.calcHist([frame], [0], None, [256], [0, 256])
	# Plot le histogramme noir et blanc
	plt.figure(num=10)
	plt.clf()
	plt.title("Grayscale Histogram")
	plt.xlabel("Grayscale value")
	plt.ylabel("Quantité de pixels")
	plt.xlim([0, 256])
	plt.plot(gray_hist, 'black', linewidth = 0.5)
	plt.draw()
	plt.pause(1e-3)
	return gray_hist
	
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

def plot_comparaison(Met1,Met2, Met3, img):
	plt.figure(num=img, figsize=(4, 4))
	plt.clf()
	plt.plot(Met1, 'b', linewidth = 1, label = "Méthode 1: BW_hist")
	plt.plot(Met2, 'red', linewidth = 1, label = "Méthode 2: HOG_hist")
	plt.plot(Met3, 'lime', linewidth = 1, label = "Méthode 3: BW_HOG")
	plt.legend(loc='upper left', fontsize='small', frameon=False)
	plt.title("Détection de changement de plans")
	plt.xlabel("Numero de frames")
	plt.ylabel("Similiarité entre les frames $F_N$ et $F_{N-1}$")
	plt.draw()
	plt.pause(0.0001)

def plot_hist2D(bwHog_hist, number_ori):
	## Histogramme avec la probabilité jointe
	plt.figure(num='Histogramme 2D')
	plt.clf()
	plt.imshow(bwHog_hist,interpolation = 'nearest', aspect='auto')
	plt.title("Hisogramme 2D - Combinaison")
	plt.ylim([0, number_ori-1])
	plt.xlabel("Histogramme en échelle de gris")
	plt.ylabel("HOG")
	plt.colorbar()
	plt.draw();
	
