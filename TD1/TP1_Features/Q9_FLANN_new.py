import numpy as np
import cv2

from matplotlib import pyplot as plt
from numpy import shape
import math
import sys

from PIL import Image


class Transform_par:
	deg = 0
	trans_x = 0
	trans_y = 0
	resize = 0
	trans_type = 0
	epsilon = 1.5


def image_transformation(img1, transform_par):
	
	if transform_par.trans_type == 2: #Rotation
		M = cv2.getRotationMatrix2D((img1.shape[0]/2,img1.shape[1]/2), transform_par.deg,1)
		img2 = cv2.warpAffine(img1, M, (img1.shape[0], img1.shape[1]))
		return img2
	

	if transform_par.trans_type == 3: #Resize
		width = int(img1.shape[1] * transform_par.resize / 100)
		height = int(img1.shape[0] * transform_par.resize / 100)
		dim = (width, height)
		# resize image
		img2 = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA) 
		return img2
	
def point_transformation(img1, List_I1Coordinates, transform_par):
	### Valeur de l'erreur acceptable
	count = 0
	### Initialises les valeurs des points d'interet de limage originale (I1) et l'image transformé (I2)
	list_qCoordinates = []
	for i in range (len(List_I1Coordinates)):
		x1 = List_I1Coordinates[i][0]
		y1 = List_I1Coordinates[i][1]

		
		#### Transforme le point d'interet vers l'image 
		if transform_par.trans_type == 2: #Rotation
			## Define the center of the image
			ox, oy = img1.shape[0]/2, img1.shape[1]/2

			# Rotation des points d'interets de l'image originale
			qx = ox + math.cos(transform_par.deg*math.pi/180) * (x1 - ox) + math.sin(transform_par.deg*math.pi/180) * (y1 - oy)
			qy = oy + -math.sin(transform_par.deg*math.pi/180) * (x1 - ox) + math.cos(transform_par.deg*math.pi/180) * (y1 - oy)
			list_qCoordinates.append([qx,qy])
		if transform_par.trans_type == 3: #Resize
			qx = x1*transform_par.resize/100
			qy = y1*transform_par.resize/100
			list_qCoordinates.append([qx,qy])
	return list_qCoordinates
	
	
def calcule_quality(list_qCoordinates, List_I2Coordinates, transform_par):
	### Valeur de l'erreur acceptable
	count = 0
	for i in range (len(list_qCoordinates)):
		qx = list_qCoordinates[i][0]
		qy = list_qCoordinates[i][1]
		x2 = List_I2Coordinates[i][0]
		y2 = List_I2Coordinates[i][1]

		## Calcule la distance entre le points d'interets de l'image originale et l'image transformé, et increment la variable de count
		distance = (abs(qx-x2)**2+abs(qy-y2)**2)**0.5
		if distance < transform_par.epsilon:
			count += 1
	return (count/len(List_I1Coordinates))*100
	
def calcul_keypoints(img1,img2):
	#Début du calcul
	t1 = cv2.getTickCount()
	#Création des objets "keypoints"
	if detector == 1:
	  kp1 = cv2.ORB_create(nfeatures = 500,#Par défaut : 500
						   scaleFactor = 1.2,#Par défaut : 1.2
						   nlevels = 8)#Par défaut : 8
	  kp2 = cv2.ORB_create(nfeatures=500,
							scaleFactor = 1.2,
							nlevels = 8)
	  print("Détecteur : ORB")
	else:
	  kp1 = cv2.KAZE_create(upright = False,#Par défaut : false
						threshold = 0.001,#Par défaut : 0.001
					nOctaves = 4,#Par défaut : 4
					nOctaveLayers = 4,#Par défaut : 4
					diffusivity = 2)#Par défaut : 2
	  kp2 = cv2.KAZE_create(upright = False,#Par défaut : false
					threshold = 0.001,#Par défaut : 0.001
					nOctaves = 4,#Par défaut : 4
					nOctaveLayers = 4,#Par défaut : 4
					diffusivity = 2)#Par défaut : 2
	  print("Détecteur : KAZE")
  
  
  
	#Conversion en niveau de gris
	gray1 =  cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
	gray2 =  cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)


	#Détection et description des keypoints
	pts1, desc1 = kp1.detectAndCompute(gray1,None)
	pts2, desc2 = kp2.detectAndCompute(gray2,None)

	t2 = cv2.getTickCount()
	time = (t2 - t1)/ cv2.getTickFrequency()
	print("Détection points et calcul descripteurs :",time,"s")



	# Calcul de l'appariement
	t1 = cv2.getTickCount()

	# Paramètres de FLANN 
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks=50) 

	flann = cv2.FlannBasedMatcher(index_params,search_params)

	matches = flann.knnMatch(np.asarray(desc1, np.float32),np.asarray(desc2, np.float32),k=2)


	# Application du ratio test
	good = []
	for m,n in matches:
		if m.distance < 0.7*n.distance:
			good.append([m])
	t2 = cv2.getTickCount()
	time = (t2 - t1)/ cv2.getTickFrequency()
	print("Calcul de l'appariement :",time,"s")
	
	print('The number of appairements is %i' %len(matches))


	## Initialize lists pour sauvegarder les points d'interet
	list_kp1 = []
	list_kp2 = []

	
	for (mat,m2) in matches:
		# Obtenez les keypoints correspondants pour chacune des images
		img1_idx = mat.queryIdx
		img2_idx = mat.trainIdx
		
		# Obtenez les coordonnées dans le bon ordre pour les coordonnées des keypoints
		(x1,y1) = pts1[img1_idx].pt
		(x2,y2) = pts2[img2_idx].pt

		# Append les valeurs de x et y a les lists
		list_kp1.append((x1, y1))
		list_kp2.append((x2, y2))
	# ~ print ("---------kdkdkdkdkd", list_kp1)
	return list_kp1, list_kp2, matches, pts1, pts2, gray1, gray2, good

if len(sys.argv) != 2:
  print ("Usage :",sys.argv[0],"detector(= orb ou kaze)")
  sys.exit(2)
if sys.argv[1].lower() == "orb":
  detector = 1
elif sys.argv[1].lower() == "kaze":
  detector = 2
else:
  print ("Usage :",sys.argv[0],"detector(= orb ou kaze)")
  sys.exit(2)

#Lecture de la paire d'images
img1 = cv2.imread('../Image_Pairs/torb_small1.png')
print("Dimension de l'image 1 :",img1.shape[0],"lignes x",img1.shape[1],"colonnes")
print("Type de l'image 1 :",img1.dtype)


################### Transformation de l'image originale
transform_par = Transform_par()		#Initialisé à 0
transform_par.deg = 40
transform_par.trans_x = 0
transform_par.trans_y = 0
transform_par.resize = 150
transform_par.trans_type = 2 		#Options:	2 = Rotation, 3 = Resize

### Transformation1: Rotation
img2 = image_transformation(img1, transform_par) #Transformation de l'image
List_I1Coordinates, List_I2Coordinates, matches, pts1, pts2, gray1, gray2, good = calcul_keypoints(img1, img2)	#Calcul des point d'interets
list_qCoordinates = point_transformation(img1, List_I1Coordinates, transform_par)  #Calcule la qualité de les appariements
quality_factor = calcule_quality(list_qCoordinates, List_I2Coordinates, transform_par)
print("Qualité des appariements: %.2f%%" % (quality_factor))	



# Affichage
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   flags = 0)

# Affichage des appariements qui respectent le ratio test
img3 = cv2.drawMatchesKnn(gray1,pts1,gray2,pts2,good,None,**draw_params)

Nb_ok = len(good)
plt.imshow(img3),plt.title('%i appariements OK'%Nb_ok)
plt.show()

print('Le numero d appariements OK sur l image originelle est %i' %Nb_ok)

print('Le rapport entre les appariements bons et le numero total d appariements est %f' %(Nb_ok/len(matches)))


