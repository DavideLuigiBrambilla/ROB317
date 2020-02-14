import cv2
import numpy as np
from Q4_functions import *

# ~ ##Reads the video from file
capture = cv2.VideoCapture("TP2_Videos/Extrait5-Matrix-Helicopter_Scene(280p).m4v")
# ~ ## Reads video from webcam
# ~ capture = cv2.VideoCapture(0)

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
VxVyHist_old = np.zeros_like(Calcule_Flow_histogramme(flow))
hog_hist = apply_hog(frame2, number_ori)

## Counter pour le frames
countFrames = 0

## Variables utilisées pour la détéction de plan
plans = []
plans_count = 0

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
	@ Threshold: 0.85
	"""
	## Sauvegard l'ancien et le nouveau histogramme
	yuvHist_old = yuvHist
	yuvHist = Calcule_2D_YUV_histogramme(ret, frame2)
	yuv_toCorr = cv2.compareHist(yuvHist_old, yuvHist, cv2.HISTCMP_CORREL)
	if yuv_toCorr<0.85:
		detec_yuv = 1
		
	"""
	@ METHODE 2: (Vx,Vy)Histogramme
	Seuil: 0.18
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
	Seuil: 0.95
	"""	
	# ~ ### Calcule le HOG
	hog_hist_old = hog_hist
	frame2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY) 
	hog_hist = apply_hog(frame2, number_ori)
	hog_hist_toCorr = np.corrcoef(hog_hist_old, hog_hist)[0,1]
	if hog_hist_toCorr<0.95:
		detec_hog = 1
		
	## Ponderation pour détécter le changement de plan	
	alpha = 1
	beta = 1
	gamma = 1
	detec_combine = alpha*detec_yuv + beta*detec_flow + gamma*detec_hog
	
	## Vote majoritaire pour la détéction de plans
	if detec_combine>1:
		if plans_count==0:
			plans.append([0,countFrames])
			# ~ teste = countFrames
			print ("Plan détécté entre les frames: ", plans[plans_count])
			plans_count+=1
			# ~ plans.append([0,countFrames])
		else:
			plans.append([plans[plans_count-1][1]+1, countFrames])
			print ("Plan détécté entre les frames: ", plans[plans_count] )
			plans_count+=1
	countFrames+=1

	# Type "q" to quit
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	prvs = next
	ret, frame2 = capture.read()
	if (ret):
		next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY) 

cv2.destroyAllWindows()
