import cv2
import matplotlib.pyplot as plt
import numpy as np
from Q1_functions import *
from time import sleep

## Lecture du video a paritr d'un fichier
capture = cv2.VideoCapture("TP2_Videos/Extrait4-Entracte-Poursuite_Corbillard(358p).m4v")
## Lecture de la camera
# ~ capture = cv2.VideoCapture(0)


## Prendre l'image actuel e l'image suivante
(ret, frameN) = capture.read() # Passe à  l'image suivante
(ret, frameN_moins1) = capture.read()

## Conversion de l'image BW-3channels en image BW 1 channel
frameN = cv2.cvtColor(frameN, cv2.COLOR_BGR2GRAY)
frameN_moins1 = cv2.cvtColor(frameN_moins1, cv2.COLOR_BGR2GRAY)

## Initialise les variables auxiliares
## Programme counter
count_hist = 0

## Quantité de orientations pour le HOG
number_ori = 12

## Listes pour le plot
sim_bwHist = []
sim_Hog = []
sim_bwHog = []
sim_bwHist4 = []

## Histogrammes pour la comparaison
bw_hist = np.reshape(np.array(Calcule_BW_histogramme(capture)), (1,256))
hog_hist = np.reshape(apply_hog(frameN, number_ori), (number_ori,1))
bwHog_hist = np.zeros_like(np.dot(hog_hist,bw_hist))
hog_hist = np.reshape(hog_hist, (1,number_ori))

while True:	
	if not ret:
		break
		
	### Affiche le video
	cv2.namedWindow('Video en échelle de gris')        
	cv2.moveWindow('Video en échelle de gris', 500,0)  
	cv2.imshow('Video en échelle de gris', frameN_moins1)	
	
	### Calcule le histogramme de l'image en échelle de gris
	bw_hist_old = bw_hist
	bw_hist = np.reshape(np.array(Calcule_BW_histogramme(capture)), (1,256))
	## Normalisation du histogramme BW
	bw_hist[:] = bw_hist[:]**0.5
	bw_hist[:] = (bw_hist[:]/np.max(bw_hist))
	
	### Calcule le HOG
	hog_hist_old = hog_hist
	hog_hist = apply_hog(frameN_moins1, number_ori)
	## Normalisation du HOG
	hog_hist[:] = hog_hist[:]**0.5
	hog_hist[:] = (hog_hist[:]/np.max(hog_hist))
	
	
	hog_hist = np.reshape(hog_hist, (number_ori,1))
	### Calcule le Histogramme2D comme une combinaison entre le histogramme de l'image en échelle de gris et le HOG
	bwHog_hist_old = bwHog_hist
	bwHog_hist = np.dot(hog_hist,bw_hist)
	## Normalisation du HOG
	bwHog_hist[:] = bwHog_hist[:]**0.5
	bwHog_hist[:] = (bwHog_hist[:]/np.max(bwHog_hist))
	## Plot Histogramme 2D
	plot_hist2D(bwHog_hist, number_ori)
	
	hog_hist = np.reshape(hog_hist, (1,number_ori))
	

	## Initialisation des variables pour vérifiér la similarité
	sim_bwHist_norm = 0
	sim_Hog_norm = 0
	sim_bwHog_norm = 0
	
	if (count_hist!=0):
		## Calcule la similarité entre les mesure pour l'image N e l'image N-1
		bw_comp = np.corrcoef(bw_hist, bw_hist_old)[0,1]
		hog_hist_comp = np.corrcoef(hog_hist_old, hog_hist)[0,1]
		hist_comp = compare_images(bwHog_hist_old, bwHog_hist)
		
		## Ajoute les calculs de similarité dedans listes pour ploter
		sim_bwHist.append(1-hog_hist_comp)
		sim_Hog.append(1-bw_comp)
		sim_bwHog.append(hist_comp)
		
		# Normalisation des listes
		sim_bwHist_norm = sim_bwHist/np.max(sim_bwHist)
		sim_Hog_norm = sim_Hog/np.max(sim_Hog)
		sim_bwHog_norm = sim_bwHog/np.max(sim_bwHog)

		## Plots le graphique de comparaison entre les images
		plot_comparaison(sim_bwHist_norm,sim_Hog_norm,sim_bwHog_norm,1)
	count_hist+=1
		
	##s Type "q" to quit
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	
	frameN = frameN_moins1
	ret, frameN_moins1 = capture.read()
	frameN_moins1 = cv2.cvtColor(frameN_moins1, cv2.COLOR_BGR2GRAY)

sleep(5)
