import os
import os.path
import glob
import sys
import numpy as np
from scipy.stats import norm
from astropy.stats import median_absolute_deviation as mad
from astropy.io import fits
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import pickle

def I_tph(x, A, tau):
	return A*(np.exp(-x/tau)-np.exp(-2*x/tau))

masks_dir="masks_thr5"

ntimes_dip_low=3	# Threshold (dipoles that repeat more than ntimes_dip_low are extracted)
ntimes_dip_up=20	# Threshold (dipoles that repeat less than ntimes_dip_up are extracted)

if len(sys.argv) > 1:
	files = sys.argv[1:]
	files.sort(key=os.path.getmtime)

	dirname=os.path.dirname(files[0])			# Get dirname of first element in files

	#-----Check if mask_Ndipoles exists and extract dipoles-----

	mask_Ndipoles_img=dirname+"/"+masks_dir+"/mask_Ndipoles.fits"
	if os.path.exists(mask_Ndipoles_img):
		hdul_Ndipoles=fits.open(mask_Ndipoles_img)
		rows_dipoles=np.array([], dtype=np.int8)
		cols_dipoles=np.array([], dtype=np.int8)
		exts_dipoles=np.array([], dtype=np.int8)

		for i in range(len(hdul_Ndipoles)):
			header_Ndipoles = hdul_Ndipoles[i].header
			data_Ndipoles = hdul_Ndipoles[i].data
#			ntimes_dip_low = np.amax(data_Ndipoles)
			print(data_Ndipoles.shape)
			totpix = float(header_Ndipoles['TOTPIX_TPUMP'])
			if np.amax(data_Ndipoles)>0:
				rows, cols = np.nonzero((data_Ndipoles>ntimes_dip_low) & (data_Ndipoles<ntimes_dip_up))
#				print(rows, cols)
				rows_dipoles=np.append(rows_dipoles, rows)
				cols_dipoles=np.append(cols_dipoles, cols)
				exts_dipoles=np.append(exts_dipoles, np.full_like(rows, i))
#		print(rows_dipoles, cols_dipoles, exts_dipoles)
		list_ext=np.unique(exts_dipoles)
		list_ext=[0]
		print(list_ext, len(list_ext))
		totexts = len(list_ext)

		#-----Loop over images-----
		list_intensities=[]
		coef_PcD=[]
		coef_tau=[]

		for image in files:
			hdul=fits.open(image)
#			hdul.info()
			img=os.path.basename(image)			# Get basename of image
			print(img)
			intensities_tph=[]

			mask_img=dirname+"/"+masks_dir+"/mask_"+img
			if os.path.exists(mask_img):			# Check if mask exists
#				print(mask_img)
				hdul_mask=fits.open(mask_img)
#				maskctr=0

				for i in list_ext:
					data=hdul[i].data		# Load data
					header=hdul[i].header		# Load header
					dtph=header['HIERARCH DELAY_TPH']	# Extract info from header

					if data is not None:				# Check if data is not empty
						data_mask = hdul_mask[i].data
						data_mask = data_mask.astype(int)

						index_dip = np.argwhere(exts_dipoles==i).flatten()

						#-----Loop over dipoles-----

						for dip in index_dip:
							if data_mask[rows_dipoles[dip], cols_dipoles[dip]]!=0:
								intensity_dip = abs((data[rows_dipoles[dip]+1, cols_dipoles[dip]]) - data[rows_dipoles[dip], cols_dipoles[dip]])/2
							else:
								intensity_dip = np.nan
#							intensity_dip = abs((data[rows_dipoles[dip]+1, cols_dipoles[dip]]) - data[rows_dipoles[dip], cols_dipoles[dip]])/2
							intensities_tph.append(intensity_dip)

#						print(np.nonzero(np.bitwise_and(data_mask, np.full_like(data_mask, 1))))
#						print(np.nonzero(data_mask))
#						print(np.argwhere(data_mask>0))

#						data=data*data_mask
#						dipoles=np.argwhere(data_mask==2)
#						print(dipoles)
#						for item in dipoles:
#							print(data[item])
					
#						print(datanp.argwhere(data>0))
#						maskctr=maskctr+1
				intensities_tph.append(int(dtph))
			else:
				print(mask_img, "does not exist! Exiting.")
				exit()

			list_intensities.append(intensities_tph)
		arr_intensities=np.array(list_intensities)

		#-----Plot each dipole-----
		for dip in range(len(arr_intensities.T)):
			y = arr_intensities.T[dip][np.isfinite(arr_intensities.T[dip])]
			x = arr_intensities.T[-1][np.isfinite(arr_intensities.T[dip])]
			x_fit = np.linspace(0, max(x), 50000)

			try:
				coef_1,cov_1 = curve_fit(I_tph, x, y, p0=[max(y), x[y.argmax()]], maxfev=1000)
#				coef_1,cov_1 = curve_fit(I_tph, arr_intensities.T[-1], arr_intensities.T[dip], p0=[max(arr_intensities.T[dip]), arr_intensities.T[-1][arr_intensities.T[dip].argmax()]])

				y_fit = I_tph(x, *coef_1)
#				y_fit = I_tph(arr_intensities.T[-1], *coef_1)
				ss_res = np.sum((y - y_fit) ** 2)							# Residual sum of squares
#				ss_res = np.sum((arr_intensities.T[dip] - y_fit) ** 2)					# Residual sum of squares
				ss_tot = np.sum((y - np.mean(y)) ** 2)							# Total sum of squares
#				ss_tot = np.sum((arr_intensities.T[dip] - np.mean(arr_intensities.T[dip])) ** 2)	# Total sum of squares
				r2 = 1 - (ss_res / ss_tot)	    							# R-squared

				if r2>0.50:
#					plt.plot(x, y, ".")
#					plt.plot(arr_intensities.T[-1], arr_intensities.T[dip], ".")
#					plt.plot(x_fit,I_tph(x_fit,*coef_1),color='red',label='Fit')
#					plt.show()
					coef_PcD.append(coef_1[0]/(40000*200))
					coef_tau.append(coef_1[1])
			except:
				pass

	else:
		print("Ndipoles mask file does not exist! Exiting.")
		exit()

	with open(dirname+"/"+masks_dir+"/coef_tau.pkl", 'wb') as f:
		pickle.dump(coef_tau, f)

	plt.hist(np.array(coef_tau)*1e-6/15, bins=np.logspace(-7,5,100), histtype='step', color='r', weights=np.ones_like(np.array(coef_tau))/(totexts*totpix))
	plt.grid()
	plt.ylabel('Number of traps / pix')
	plt.xlabel('Tau (s)')
	plt.xscale('log')
	plt.yscale('log')
	plt.ylim([0.00001, 0.01])
	plt.savefig(dirname+"/"+masks_dir+'/hist_Ndipoles_ext'+str(list_ext)+'.png')
	plt.show()

else:
	print("To run do: python dipoles_profile.py path/img*.fits")
