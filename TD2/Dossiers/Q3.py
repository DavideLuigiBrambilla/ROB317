import cv2
import matplotlib.pyplot as plt
import numpy as np
from Q3_functions import *

#Ouverture du flux video
# cap = cv2.VideoCapture("TP2_Videos/Extrait3-Vertigo-Dream_Scene(320p).m4v")
cap = cv2.VideoCapture(0)

ret, frame1 = cap.read() # Passe à  l'image suivante
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY) # Passage en niveaux de gris

ret, frame2 = cap.read()
next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY) 

## Initialise les variables pour la comparations des histogrammes	
flow = Find_flow(prvs,next)
hist_N = np.zeros_like(Calcule_Flow_histogramme(flow))
hist_N_moins1 = np.zeros_like(Calcule_Flow_histogramme(flow))
corr = []	
index = 1

while(ret):
	index += 1
	## Prendre le flow de l'image
	flow = Find_flow(prvs,next)
	
	## Compare les histogrammes								
	hist_N_moins1 = hist_N
	hist_N = Calcule_Flow_histogramme(flow)
	corr.append(cv2.compareHist(hist_N_moins1, hist_N, cv2.HISTCMP_CORREL))
	
	## Calcule la correlation entre les imagems
	plot_correlation(corr)

	## Affiche la video
	cv2.imshow('Video utilisé',frame2)
	k = cv2.waitKey(15) & 0xff
	if k == 27:
		break
	prvs = next
	ret, frame2 = cap.read()
	if (ret): ## Si il y a une image a être analysé
		next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY) 

cap.release()
cv2.destroyAllWindows()
