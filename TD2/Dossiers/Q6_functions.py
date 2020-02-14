import numpy as np
import cv2
from skimage.feature import hog
from numpy import shape
import matplotlib.pyplot as plt


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
	hist = cv2.calcHist([flow], [1, 0], None, [bins]*2, [-bins, bins]*2)	
	
	## Elimination des valeurs statics (pas importantes dans ce cas)
	# hist[hist[:,:]>np.std(hist)/2] = np.mean(hist)
	hist[hist[:,:]>np.std(hist)/2] = np.std(hist)/2
	
	## Normalisation non-lineaire
	# hist[:,:] = (hist[:,:])**0.5
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
	
def flow_hist(flow,step):
	mag_total, ang_total = cv2.cartToPolar(flow[:,:,0], flow[:,:,1]) # Conversion cartÃ©sien vers polaire
	ang_total  = (ang_total*180)/(np.pi) # Teinte (codÃ©e sur [0..179] dans OpenCV) <--> Argument
	
	ang_total = np.reshape(ang_total, shape(ang_total)[0]*shape(ang_total)[1])
	mag_total = np.reshape(mag_total, shape(mag_total)[0]*shape(mag_total)[1])

	angle = 0
	# step=15
	step_angle = 360/step
	flow_mag_hist = []
	flow_angle_hist = []
	flow_hist_descript = []
	for i in range (step):
		angle_aux = ang_total[ang_total[:]>=angle]
		mag_aux = mag_total[ang_total[:]>=angle]
		if i==step-1:
			angle_int = angle_aux[angle_aux[:]<=(angle+step_angle)]
			mag_int = mag_aux[angle_aux[:]<=(angle+step_angle)]
		else:
			angle_int = angle_aux[angle_aux[:]<(angle+step_angle)]
			mag_int = mag_aux[angle_aux[:]<(angle+step_angle)]
		
		flow_mag_hist.append(np.sum(mag_int))
		flow_angle_hist.append(shape(angle_int)[0])
		
		# print(angle, angle+step_angle,"---",mag_sum, shape(mag_total)[0], "//", shape(angle_int)[0], shape(ang_total)[0])#, ":", angle_aux_sum)
		
		# hist_flow_ori.append([mag_total, ang_total])
		
		angle += step_angle
	flow_hist_descript.append(flow_mag_hist)
	flow_hist_descript.append(flow_angle_hist)
	return flow_hist_descript

def Classification_type_plan(img, compare_angMag):
	img[img[:,:]==1] = 0
	mean_img = np.mean(img)
	std_img = np.std(img)
	ratio_img = std_img/mean_img

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
	seuil_zoom = 40
	seuil_rot_var = 0.11

	# print ("RotationZoom seuil: ", compare_angMag)
	# print (sum_H[0]+sum_H[0]+sum_H[0]+sum_H[0])
	# print ("Variance: ", mean_img)
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
	elif (sum_H[0]+sum_H[0]+sum_H[0]+sum_H[0]>seuil_zoom):
		if (compare_angMag <= seuil_rot_var):
			print("    > Type: Rotation (13)")
		else:
			print("    > Type: Zoom (14)")		
	else:
		print("    > Type: Plan fixe (15)")
