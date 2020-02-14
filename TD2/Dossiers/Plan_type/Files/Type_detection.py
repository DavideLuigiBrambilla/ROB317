
import numpy as np
import matplotlib.pyplot as plt
from numpy import shape
import cv2

name = "Zoom_3"
img = np.loadtxt(name + ".txt")
axis = np.reshape(np.ones(64)*32, 64)
indices = np.reshape(np.indices((1, 64))[1],64)






# print(np.mean(img), np.std(img), np.std(img)/np.mean(img))

# print(shape(img))

mag, ang = cv2.cartToPolar(flow[:,:,0], flow[:,:,1]) # Conversion cartÃ©sien vers polaire
ang_total  = (ang*180)/(2*np.pi) # Teinte (codÃ©e sur [0..179] dans OpenCV) <--> Argument
mag_total = (mag*255)/np.amax(mag) # Valeur <--> Norme 

img[img[:,:]==1] = 0
mean_img = np.mean(img)
std_img = np.std(img)
ratio_img = std_img/mean_img

print("Ratio:", ratio_img)
print("Variance:", std_img)
print("Mean:", mean_img)

sum_H = []
H0 = img[0:32, 0:32]
H1 = img[0:32, 32:64]
H2 = img[32:64, 0:32]
H3 = img[32:64, 32:64]
sum_H.append(np.sum(H0))
sum_H.append(np.sum(H1))
sum_H.append(np.sum(H2))
sum_H.append(np.sum(H3))

seuil_H1_H3 = 2
seuil_H0_H1 = 2
seuil_H0_H2 = 2
seuil_H2_H3 = 2
seuil_rot = 80
seuil_inner = 4
seuil_zoom = 50
seuil_rot_var = 6

if (sum_H[1] + sum_H[3]) > seuil_H1_H3*(sum_H[0] + sum_H[2]):
	if (sum_H[1] > seuil_inner*sum_H[3]):
		print("    > Type: Travelling/ tilt vers le coin inférieur gauche (01)")
	elif (sum_H[3] > seuil_inner*sum_H[1]):
		print("    > Type: Travelling/ tilt vers le coin supérieur gauche (02)")
	else:
		print("    > Type: Travelling/ tilt vers la gauche (03)")
elif (sum_H[0] + sum_H[1]) > seuil_H0_H1*(sum_H[2] + sum_H[3]):
	if (sum_H[1] > seuil_inner*sum_H[0]):
		print("    > Type: Travelling/ tilt vers le coin inférieur gauche (04)")
	elif (sum_H[0] > seuil_inner*sum_H[1]):
		print("    > Type: Travelling/ tilt vers le coin inférieur droite (05)")
	else:
		print("    > Type: Travelling/ tilt vers le bas (06)")
elif (sum_H[0] + sum_H[2]) > seuil_H0_H2*(sum_H[1] + sum_H[3]):
	if (sum_H[0] > seuil_inner*sum_H[2]):
		print("    > Type: Travelling/ tilt vers le coin inférieur droite (07)")
	elif (sum_H[2] > seuil_inner*sum_H[0]):
		print("    > Type: Travelling/ tilt vers le coin supérieur droite (08)")
	else:
		print("    > Type: Travelling/ tilt vers la droite (09)")
elif (sum_H[2] + sum_H[3]) > seuil_H2_H3*(sum_H[0] + sum_H[1]):
	if (sum_H[2] > seuil_inner*sum_H[3]):
		print("    > Type: Travelling/ tilt vers le coin supérieur droite (10)")
	elif (sum_H[3] > seuil_inner*sum_H[2]):
		print("    > Type: Travelling/ tilt vers le coin supérieur gauche (11)")
	else:
		print("    > Type: Travelling/ tilt vers le haut (12)")
elif (ratio_img < seuil_rot_var):
	print("    > Type: Rotation (13)")
elif (ratio_img >= seuil_rot_var ):
	print("    > Type: Zoom (14)")
else:
	print("    > Type: Plan fixe (15)")



## Histogramme avec la probabilité jointe
plt.figure(num="Histogram pour le "+name)
plt.title("Histogramme pour le "+ name)
plt.xlabel("Composante $V_x$")
plt.ylabel("Composante $V_y$")
plt.plot(axis,indices, "white", linewidth = 0.5)
plt.plot(indices,axis, "white", linewidth = 0.5)
plt.imshow(img,interpolation = 'nearest')
plt.colorbar()
plt.savefig(name + '.png')
plt.show()
plt.pause(0.0001)
