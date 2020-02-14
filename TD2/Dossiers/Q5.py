import cv2
import numpy as np
from Q5_functions import *

## Reads the video from file
capture = cv2.VideoCapture("TP2_Videos/Extrait5-Matrix-Helicopter_Scene(280p).m4v")
## Reads video from webcam
# capture = cv2.VideoCapture(0)

ret, frame1 = capture.read() # Passe à  l'image suivante
ret, frame2 = capture.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY) # Passage en niveaux de gris
next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY) 

## Initialise les variables pour la comparations des histogrammes	
yuvHist = np.zeros_like(Calcule_2D_YUV_histogramme(ret, frame2))
yuvHist_old = np.zeros_like(Calcule_2D_YUV_histogramme(ret, frame2))


## Quantité de orientations pour le HOG
number_ori = 8

## Initialise les variables pour la comparations des histogrammes	
flow = Find_flow(prvs,next)
VxVyHist = np.zeros_like(Calcule_Flow_histogramme(flow))
hog_hist = apply_hog(frame2, number_ori)
plan_selection = []
hog_accumulate = np.zeros_like(apply_hog(frame2, number_ori))

## Counter pour le frames
countFrames = 0

## Variables utilisées pour la détéction de plan
plans_count = 0
plans_selection = []
corrPlan_list = []

while True:
	if not ret:
		break
	## Affiche la video
	show_video(frame2)
	
	detec_yuv = 0
	detec_flow = 0
	detec_hog = 0

	"""
	@ METHODE 1: YUV Histogramme
	@ Threshold:	Extrait 1: 0.85
	@ 				Extrait 2:   -
	@ 				Extrait 3:   -
	@ 				Extrait 4:   -
	@ 				Extrait 5: 0.84
	"""
	## Sauvegard l'ancien et le nouveau histogramme
	yuvHist_old = yuvHist
	yuvHist = Calcule_2D_YUV_histogramme(ret, frame2)
	yuv_toCorr = cv2.compareHist(yuvHist_old, yuvHist, cv2.HISTCMP_CORREL)
	if yuv_toCorr<0.85:
		detec_yuv = 1
		
	"""
	@ METHODE 2: (Vx,Vy)Histogramme
	@ Threshold:	Extrait 1: 0.2
	@ 				Extrait 2:  -
	@ 				Extrait 3:  -
	@ 				Extrait 4:  -
	@ 				Extrait 5: 0.2
	"""
	## Calcule le flow de l'image
	flow = Find_flow(prvs,next)
	## Compare les histogrammes								
	VxVyHist_old = VxVyHist
	VxVyHist = Calcule_Flow_histogramme(flow)
	flow_toCorr = cv2.compareHist(VxVyHist_old, VxVyHist, cv2.HISTCMP_CORREL)
	if flow_toCorr<0.18:
		detec_flow = 1
	
	
	"""
	@ METHODE 3: HOG
	@ Threshold:	Extrait 1: 0.95
	@ 				Extrait 2: 0.95
	@ 				Extrait 3: 0.973
	@ 				Extrait 4:   -
	@ 				Extrait 5: 0.9
	"""	
	### Calcule le HOG
	hog_hist_old = hog_hist
	plan_selection.append([frame2, hog_hist])
	frame2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY) 
	hog_hist = apply_hog(frame2, number_ori)
	hog_hist_toCorr = np.corrcoef(hog_hist_old, hog_hist)[0,1]
	if hog_hist_toCorr<0.9:
		detec_hog = 1
		

	"""
	@ DÉTECTION DE PLAN E IMAGE CLEF
	@ 	1. Détéction: fait avec la fonction alpha*detec_yuv + beta*detec_flow + gamma*detec_hog,
	@		ou alpha, beta et gamma sont des coeficients de pondération pour chaque méthode de
	@		détéction
	@	2. Image Clef: Fait en utilisant le HOG pour chaque plan
	"""	
	## Ponderation pour détécter le changement de plan	
	alpha = 1
	beta = 1
	gamma = 1
	detec_combine = alpha*detec_yuv + beta*detec_flow + gamma*detec_hog
	
	## Accumulateur pour faire obtenir la valeur moyenne de HOG (réprésentatif)
	hog_accumulate += hog_hist
	
	
	## Vote majoritaire pour la détéction de plans
	if detec_combine>1:
		if plans_count==0:
			plans_selection.append([0,countFrames])
		else:
			plans_selection.append([plans_selection[plans_count-1][1]+1, countFrames])
		print (plans_count, "- Plan détécté entre les frames: ", plans_selection[plans_count] )
		plans_count+=1
		
		## Sélection d'une image clef pour le plan détecté
		hog_plan_moyen = hog_accumulate/shape(plan_selection)[0]
		for index in range (shape(plan_selection)[0]):
			corr_plan = np.corrcoef(plan_selection[index][1], hog_plan_moyen)[0,1]
			corrPlan_list.append(corr_plan)
			
		## Image clé sera celle ayant plus de similarité avec le HOG moyen
		index_cle = np.where(corrPlan_list == np.max(corrPlan_list))[0]
		
		## Sauvegard l'image clef pour chaque plan
		cv2.imwrite('Image_clef/Plan_%d.png'%plans_count,plan_selection[int(index_cle)][0])

		## Affichage de l'image clef
		cv2.imshow('Image Clef', plan_selection[int(index_cle)][0])

		### Reinitialise les lists de sélection
		plan_selection.clear()
		corrPlan_list.clear()
		hog_accumulate=0
	
	countFrames+=1
	
	# Type "q" to quit
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	prvs = next
	ret, frame2 = capture.read()
	if (ret):
		next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY) 

cv2.destroyAllWindows()
