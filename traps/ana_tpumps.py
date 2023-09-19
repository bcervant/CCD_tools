import os
import glob
import sys
import numpy as np
from scipy.stats import norm
from astropy.stats import median_absolute_deviation as mad
from astropy.io import fits
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

scale_thr=5		# Variable to determine level of threshold
ntimes_dipole=15	# Threshold (dipoles that repeat more than ntimes_dipole are extracted)

# Image dimensions
nrow=520	#650		
ncol=420	#700
xOS=350		#3080	538

# Area of interest
xmin=10; xmax=xOS; ymin=5; ymax=510

numext=4	
list_ext=[0, 1, 2, 3]	# Extensions go from 0 to 3 in proc*.fits
totpix=(ymax-ymin+1)*(xmax-xmin+1)

if len(sys.argv) > 1:
	files = sys.argv[1:]
	files.sort(key=os.path.getmtime)

	dirname=os.path.dirname(files[0])	# Get dirname of first element in files

	overscan_mask = np.s_[:, xOS:]		# 538 <= x
	mask=np.s_[ymin-1:ymax, xmin-1:xmax]	# Area where variable will be computed

	list_dtph=[]
	list_ndipoles=[]

	list_xlabels=[]

	img_Ndipoles=np.zeros((numext, nrow, ncol))		# Create empty image to store number of times a dipole was found

	masks_dir=dirname+"/masks_thr"+str(scale_thr)		# Directory to store the created masks
	os.makedirs(masks_dir, exist_ok = True) 

	for image in files:

		print(image)
		hdul=fits.open(image)
#		hdul.info()
		img=os.path.basename(image)				# Get basename of image
#		list_ext=range(1, len(hdul))

#		list_xlabels.append(img.split("module12_")[1].split("_wledon")[0])	# X labels of histogram

		mask_img = fits.HDUList()

		ndipoles_img=[]

		for i in range(numext):
			mask_dipoles_ext_low=np.zeros((nrow, ncol))
			mask_dipoles_ext_up=np.zeros((nrow, ncol))

			header=hdul[i].header			# Load header
			dtph=header['HIERARCH DELAY_TPH']	# Extract info from header

			if (i in list_ext):
				data=hdul[i].data	# Load data

				if data is not None:				# Check if data is not empty
					data = data - np.median(data[overscan_mask], axis=1, keepdims=True)	# Subtract OS median per row
					data = data - np.median(data, axis=0, keepdims=True)			# Subtract median per column
					data=data[mask]

					#-----Auxiliar arrays-----
					sign=np.sign(data[1:, :])*np.sign(data[:-1, :])		# If two consecutive pixels have same sign, sign=1 (else sign=-1)
#					print(sign)

					minimum=np.minimum(abs(data[1:, :]), abs(data[:-1, :]))	# Array with the minimum element comparing two arrays
					maximum=np.maximum(abs(data[1:, :]), abs(data[:-1, :]))	# Array with the maximum element comparing two arrays
					scalar=minimum/maximum					# Dipoles should have scalar~=1
#					print(scalar)

					subdata=abs(data[1:, :]-data[:-1, :])			# Array created subtracting contiguous pixels in column pix[i+1]-pix[i] (distance between two consecutive pixels)
#					print(subdata.shape)

					subdata_dipoles=-subdata*sign*scalar
					subdata_stars=subdata*sign

#					plt.plot(data)
#					plt.plot(subdata_dipoles)
#					plt.plot(subdata_stars)
#					plt.show()

					med_subdata=np.median(subdata, axis=0, keepdims=True)	# Median of distances between two consecutive pixels per column
#					print(med_subdata.shape, med_subdata)

					img_dipoles_ext=np.zeros(data.shape)

					#-----Loop to find dipoles in each column-----
					ndipoles_ext=0
					for col in range(len(subdata_dipoles[0,:])):
						thr=scale_thr*med_subdata.flatten()[col]		# Amplitude threshold based on the median of the distances between two consecutive pixels
#						peaks_dipoles, _ = find_peaks(subdata_dipoles[:,col], height=thr)
						peaks_dipoles = np.argwhere(subdata_dipoles[:,col]>=thr).flatten()

#						plt.plot(data[:,col])
#						plt.plot(subdata_dipoles[:,col])
#						plt.axhline(y=thr, c='r', ls='--')
#						plt.plot(peaks_dipoles, subdata_dipoles[:,col][peaks_dipoles], "x")
#						plt.show()

						img_dipoles_ext[:,col][peaks_dipoles]=1			# Create image identifying dipoles (lower)
						img_dipoles_ext[:,col][peaks_dipoles+1]=1		# Create image identifying dipoles (upper)

						mask_dipoles_ext_low[:,col+xmin][peaks_dipoles+ymin]=1	# Create image identifying dipoles with image dimensions (lower)
						mask_dipoles_ext_up[:,col+xmin][peaks_dipoles+ymin+1]=2	# Create image identifying dipoles with image dimensions (upper)

						ndipoles_ext=ndipoles_ext+len(peaks_dipoles)		# Sum number of dipoles found per column to the total number of dipoles in each ext

					ndipoles_img.append(ndipoles_ext)
					img_Ndipoles[i,:,:]=img_Ndipoles[i,:,:]+mask_dipoles_ext_low

					mask_img.append(fits.ImageHDU(mask_dipoles_ext_low+mask_dipoles_ext_up, header))

					#-----Plot data VS img with dipoles per ext-----
#					fig_ext, axs_ext = plt.subplots(1, 2, figsize=(10, 5), sharey=True, sharex=True)
#					fig_ext.suptitle(img, fontsize=10)
#					axs_ext[0].imshow(data, origin='lower')
#					axs_ext[1].imshow(img_dipoles_ext, origin='lower')
#					plt.show()
			else:
				ndipoles_img.append(0)
				
				mask_img.append(fits.ImageHDU(mask_dipoles_ext_low+mask_dipoles_ext_up, header))

		list_dtph.append(int(dtph))
		list_ndipoles.append(ndipoles_img)

		mask_img.writeto(masks_dir+"/mask_"+img, overwrite=True)		# Create mask per image

		hdul.close()

	arr_dtph=np.array(list_dtph)
	arr_ndipoles=np.array(list_ndipoles)
#	print(arr_dtph)
#	print(arr_ndipoles)	
	
	if (len(files)>1):
		#-----Create mask storing the number of images where a dipole appears-----
		mask_Ndipoles_img = fits.HDUList()
		for i in range(numext):
			hdu = fits.ImageHDU(img_Ndipoles[i,:,:])
			hdu.header['TOTPIX_TPUMP'] = totpix
			mask_Ndipoles_img.append(hdu)
		mask_Ndipoles_img.writeto(masks_dir+"/mask_Ndipoles.fits", overwrite=True)

		#-----Save coordinates of dipoles that appear in multiple images-----
		coordinates=np.argwhere(img_Ndipoles>ntimes_dipole)
#		np.savetxt(dirname+'/dipolesXY_thr'+str(scale_thr)+'.txt', coordinates)	# Columns in TXT file: ext Y_dipole X_dipole

		#-----Save number of dipoles vs dtph in txt-----
		points=np.concatenate(([arr_dtph.T], arr_ndipoles.T))
		np.savetxt(dirname+'/Ndipoles_thr'+str(scale_thr)+'.txt', points.T)	# Columns in TXT file: dtph Ndipoles_ext1 Ndipoles_ext2 ...

		#-----Plot of the number of dipoles vs dtph-----
		fig_all, axs_all = plt.subplots(2, len(list_ext), figsize=(5*len(list_ext), 10), sharex='row', sharey='row')
		figctr=0
		for i in list_ext:
			axs_all[0, figctr].plot(arr_dtph, arr_ndipoles.T[i]*100/totpix, "o")
			axs_all[0, figctr].set_xscale('log')
			axs_all[0, figctr].set_yscale('log')
			axs_all[0, figctr].set_xlabel('dtph (clocks)')
			axs_all[0, figctr].set_ylabel('Ndipoles / totpix (%)')
			axs_all[0, figctr].set_title('ext '+str(i+1))
			axs_all[0, figctr].grid()
			plt.colorbar(axs_all[1, figctr].imshow(img_Ndipoles[i,:,:], origin='lower'), ax=axs_all[1, figctr])
			figctr=figctr+1
		plt.savefig(dirname+'/Ndipoles_thr'+str(scale_thr)+'.png')
		plt.show()

		#-----Histogram of the number of dipoles vs voltages-----
#		x_axis=np.arange(len(list_xlabels))
#		offset=-0.4
#		for i in range(numext):
#			plt.bar(x_axis+offset, arr_ndipoles.T[i], 0.2, label='ext '+str(i+1))
#			offset=offset+0.2
#		plt.xticks(x_axis, list_xlabels, rotation=0, fontsize=8)
#		plt.ylabel('Ndipoles')
#		plt.yscale('log')
#		plt.legend()
			#		plt.savefig(dirname+'/Ndipoles_voltages_dtph'+str(dtph)+'.png')
#		plt.show()

else:
	print("To run do: python3 plotdata.py path/img*.fits")
