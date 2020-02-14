import numpy as np
import cv2

from matplotlib import pyplot as plt

#Lecture image en niveau de gris et conversion en float64
img=np.float64(cv2.imread('../Image_Pairs/Graffiti0.png',cv2.IMREAD_GRAYSCALE))
(h,w) = img.shape
print("Dimension de l'image :",h,"lignes x",w,"colonnes")
print("Type de l'image :",img.dtype)

#Debut du calcul
t1 = cv2.getTickCount()
Theta = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
# Mettre ici le calcul de la fonction d'interet de Harris
alfa = 0.06
sigma = 5
Ix = cv2.Sobel(Theta, -1, 1, 0, ksize = sigma)          #derivation et smoothing gaussian avec un filter de Sobel dans la direction x...
Iy = cv2.Sobel(Theta, -1, 0, 1, ksize = sigma)          # ...et dans la direction y
Gkernel = cv2.getGaussianKernel(2*sigma, -1)            # definition de la deuxieme gaussienne
Ixx = cv2.filter2D(Ix**2, -1, Gkernel)                  # calcul termes matrice d'autocorrelation: moyenne local en utilisant un filtre gaussien 
Iyy = cv2.filter2D(Iy**2, -1, Gkernel)                  # moyenne local en utilisant un filtre gaussien
Ixy = cv2.filter2D(Ix*Iy, -1, Gkernel)                  # moyenne local en utilisant un filtre gaussien 
Theta = (Ixx*Iyy - Ixy**2) - alfa * (Ixx + Iyy)**2      #calcul de la fonction d'interet [det(Matrix) - alfa * trace(Matrix)]

#                                                       #comparaison avec la fonction Harris de OpenCV
# Calcul des maxima locaux et seuillage
Theta_maxloc = cv2.copyMakeBorder(Theta,0,0,0,0,cv2.BORDER_REPLICATE)
d_maxloc = 3
seuil_relatif = 0.01
se = np.ones((d_maxloc,d_maxloc),np.uint8)
Theta_dil = cv2.dilate(Theta,se)                        #premiere dilatation pour effacer les non-maxima locaux
#Suppression des non-maxima-locaux
Theta_maxloc[Theta < Theta_dil] = 0.0
#On neglige egalement les valeurs trop faibles
Theta_maxloc[Theta < seuil_relatif*Theta.max()] = 0.0
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Mon calcul des points de Harris :",time,"s")
print("Nombre de cycles par pixel :",(t2 - t1)/(h*w),"cpp")

plt.subplot(131)
plt.imshow(img,cmap = 'gray')
plt.title('Image originale')

plt.subplot(132)
plt.imshow(Theta,cmap = 'gray')
plt.title('Fonction de Harris')

se_croix = np.uint8([[1, 0, 0, 0, 1],
[0, 1, 0, 1, 0],[0, 0, 1, 0, 0],
[0, 1, 0, 1, 0],[1, 0, 0, 0, 1]])
Theta_ml_dil = cv2.dilate(Theta_maxloc,se_croix)        #deuxieme dilatation pour donner la forme a x
#Relecture image pour affichage couleur
Img_pts=cv2.imread('../Image_Pairs/Graffiti0.png',cv2.IMREAD_COLOR)
(h,w,c) = Img_pts.shape
print("Dimension de l'image :",h,"lignes x",w,"colonnes x",c,"canaux")
print("Type de l'image :",Img_pts.dtype)
#On affiche les points (croix) en rouge
Img_pts[Theta_ml_dil > 0] = [255,0,0]
plt.subplot(133)
plt.imshow(Img_pts)
plt.title('Points de Harris')

plt.show()
