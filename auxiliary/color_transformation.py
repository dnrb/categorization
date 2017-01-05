import numpy as np
A = np.array

def lab_to_xyz(lab):
	epsilon = 0.008856
	kappa = 903.3
	L,a,b = lab
	ref_white = A([95.047,100.00,108.883])
	#
	fy = (L+16)/116
	fx = a/500 + fy
	fz = fy - b/200
	#
	x = np.power(fx, 3) if np.power(fx, 3) > epsilon else (116*fx - 16)/kappa
	y = np.power((L+16)/116,3) if L > kappa * epsilon else L/kappa
	z = np.power(fz, 3) if np.power(fz, 3) > epsilon else (116*fz - 16)/kappa
	xyz = A([x,y,z]) * ref_white
	return xyz

def xyz_to_rgb(xyz):
	var_X, var_Y, var_Z = xyz/100
	var_R = var_X *  3.2406 + var_Y * -1.5372 + var_Z * -0.4986
	var_G = var_X * -0.9689 + var_Y *  1.8758 + var_Z *  0.0415
	var_B = var_X *  0.0557 + var_Y * -0.2040 + var_Z *  1.0570
	wau = 0.0031308

	if var_R > wau : var_R = 1.055 * np.power(var_R, 1 / 2.4 ) - 0.055
	else : var_R = 12.92 * var_R
	if var_G > wau : var_G = 1.055 * np.power(var_G, 1 / 2.4 ) - 0.055
	else : var_G = 12.92 * var_G
	if var_B > wau : var_B = 1.055 * np.power(var_B, 1 / 2.4 ) - 0.055
	else : var_B = 12.92 * var_B
	return np.max((np.zeros(3),A([var_R, var_G, var_B])),0)