import cv2
import matplotlib.pyplot as plt
import numpy as np

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
	
def plot_correlation(corr):
	plt.figure(num=2, figsize=(4, 4))
	plt.clf()
	plt.rcParams["figure.figsize"] = (5,5)
	plt.plot(corr, 'b', linewidth = 0.5)
	plt.ylim([0, 1])
	plt.title("Correlation des histogrammes h et h-1")
	plt.xlabel("Numero de frames")
	plt.ylabel("Correlation (%)")
	plt.draw()
	plt.pause(0.0001)	
	
def Calcule_Flow_histogramme(flow):
	bins = 64
	hist = cv2.calcHist([flow], [1, 0], None, [bins]*2, [-bins, bins]*2)	
	
	## Elimination des valeurs statics (pas importantes dans ce cas)
	hist[hist[:,:]>np.std(hist)/2] = np.std(hist)/2
	
	## Normalisation non-lineaire
	hist[:,:] = (hist[:,:]/np.max(hist))
	
	## Histogramme avec la probabilité jointe
	plt.figure(num=1)
	plt.clf()
	plt.title("Histogramme 2D des composantes $V_x$ et $V_y$")
	plt.xlabel("Composante $V_x$")
	plt.ylabel("Composante $V_y$")
	plt.imshow(hist,interpolation = 'nearest')
	plt.colorbar()
	plt.draw()
	plt.pause(1e-3)
	
	return hist
