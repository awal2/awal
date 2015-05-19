#Alex Walczak, 2015
#Measuring blinkiness of E. coli.
from __future__ import division, print_function

import cv2
import numpy as np

import sys
import itertools as it
from math import floor

from matplotlib import pyplot as plt
import matplotlib.cm as cm

import pylab as pl

from scipy.interpolate import interp1d, splrep, sproot
from scipy.signal import lfilter
import scipy.signal as signal
from scipy.ndimage.filters import median_filter
from scipy.sparse import csc_matrix, spdiags
from scipy.sparse.linalg import spsolve

from detect_peaks import detect_peaks


'''
For optimal viewing, open using IPython. Anaconda highly recommended.
When cd'ed into this script's directory at terminal, run command: 

>>> ipython notebook

To run py script in IPython (which opens automatically in browser),
>>> '%'run fname (where '%' is without quotes
					and fname is py script without .py extension)
In this case:
>>> '%'run future

'''

class Source(object):

	def __init__(self, filename):
		self.filename = filename
		self.cap = cv2.VideoCapture(filename)
		self.height = self.cap.get(4)
		self.width = self.cap.get(3)
		self.length = floor(self.cap.get(7))
		self.image_stack = np.empty((self.height, self.width, self.length))
		self.image_stack2 = np.empty((self.height, self.width, self.length, 3))
		self.bkgd_stack = np.empty((self.height, self.width, self.length))

		self.My_Zstack=None
		self.My_Covar=None
		self.My_Avg=None
		self.My_Diff=None
		self.My_Peaks=None
		self.My_Zsum=None

		self.weights=None

	# Creates a 3D matrix of the video file.
	def zstack(self):
		a=0
		while a < self.length: 
			ret, frame = self.cap.read() 
			if ret==False:
				self.cap.open(self.filename)
				print('ERROR!!!')
			gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			self.image_stack[:,:,a] = cv2.GaussianBlur(gray_img,(5,5),0) #cv2.medianBlur(gray_img,5) #gray_img #Switch to remove noise filter.
			a+=1
		self.cap.release()
		self.My_Zstack = self.image_stack
		return self.image_stack

	# Covariance of each 1D array with the one to the right in next frame.
	def covar_z(self):
		if self.My_Zstack==None:
			self.zstack()
		zstack1 = list(self.My_Zstack)
		zstack0 = list(self.My_Zstack)
		
		# Overlaps two image stacks for easy covariance.
		cur = np.delete(zstack0,0,1) #remove first column
		cur = np.delete(cur,-1,2) #del last frame
		ahead = np.delete(zstack1,-1,1) #remove last column
		ahead = np.delete(ahead,0,2) #del first frame
		mtx = np.empty((cur.shape[0],cur.shape[1]))
		wide=cur.shape[0]
		tall=cur.shape[1]
		mtx1 = np.copy(mtx)

		for i in range(0,wide):
			for j in range(0,tall):
				mtx1[i,j]=np.cov(cur[i,j],ahead[i,j])[0,1]
		self.My_Covar = mtx1

	def average_z(self):
		if self.My_Zstack==None:
			self.zstack()
		zs = np.copy(self.My_Zstack)
		mtx = np.empty((zs.shape[0],zs.shape[1]))
		for i in range(0,zs.shape[0]):
			for j in range(0,zs.shape[1]):
				mtx[i,j]=np.average(zs[i,j])
		self.My_Avg = mtx

	def sub_covar(self):
		self.My_Diff = None
		if self.My_Covar == None:
			self.covar_z()
		if self.My_Avg == None:
			self.average_z()
		self.My_Covar *= 255/self.My_Covar.max() 
		self.My_Avg *= 255/self.My_Avg.max() 
		res = self.My_Covar-self.My_Avg[:,:-1]
		res *= 255/res.max() 
		self.My_Diff = res

	def zsum(self, zstack=None):
		#Need to have made enough assmgts to call this method.
		if zstack==None:
			zs = np.copy(self.My_Zstack)
		if zstack!=None:
			zs = zstack
		mtx = np.empty((zs.shape[0],zs.shape[1]))
		for i in range(0,zs.shape[0]):
			for j in range(0,zs.shape[1]):
				mtx[i,j]=np.ndarray.sum(zs[i,j])
		self.My_Zsum = mtx
		#add max

	def bkgd_sub(self):
		i=0
		frames = self.My_Zstack
		while i < self.length:

			f = self.My_Zstack

			avg1 = np.float32(f)
			avg2 = np.float32(f)
			 

			_,f = c.read()

			cv2.accumulateWeighted(f,avg1,0.1)
			cv2.accumulateWeighted(f,avg2,0.01)

			res1 = cv2.convertScaleAbs(avg1)
			res2 = cv2.convertScaleAbs(avg2)
			display(f)
			display(res1)
			display(res2)

	def bckgd_weights(self, img, cnt_ar=None, cnt_thresh=180, thresh=17, preview=False):
		#This fxn finds how we should weight the pixels in an image to caclulate the
		#average bckgd fluorescence. We do not want to include cells in this calculation.
		#Weights is an image where cells are 0 (black), and everything else is 1s (weighted equally, white).
		#
		#cnt_ar: the array which we find cnts on. use zsum to count all cells.
		#img: from which bckgd is calc'd
		#cnt_thresh: the min pixel value from cnt_ar that will be a contour
		#thresh: threshold for what doesn't counts as bckgd.
		if cnt_ar == None:
			self.zsum()
			cnt_ar = self.My_Zsum
		cnt_ar *= 255/cnt_ar.max()
		cnt_ar = cnt_ar//1
		cnt_ar = np.array(cnt_ar,dtype='uint8')
		thim = cv2.adaptiveThreshold(cnt_ar,cnt_thresh,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,17,0)
		cnts, h = cv2.findContours(thim,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
		msk = np.zeros(img.shape,np.uint8) #zero mtx
		cv2.drawContours(msk, cnts, -1, 255, -1)
		sub_cells=cv2.bitwise_not(np.copy(img),msk.astype('float64'))*-1
		sub_cells *= 255/sub_cells.max()
		ret,thresh_img = cv2.threshold(sub_cells.astype('uint8'),thresh,255,cv2.THRESH_BINARY)
		weights=thresh_img/255

		if preview:
			print('1. Original','2. Mask','3. Cells subtracted', '4. Weights')
			display(img)
			display(msk)
			display(sub_cells)
			display(weights)

		self.weights = weights

	def adjust_zstack(self):
		#Removes bckgd fluor from ea. img.
		#trying hist eq

		self.zsum()
		zsum = self.My_Zsum
		zstk = np.copy(self.My_Zstack)

		####new:####
		for frame in range(zstk.shape[2]):
			clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
			cl1 = clahe.apply(zstk[:,:,frame].astype('uint8'))
			self.My_Zstack[:,:,frame] = cl1
		###end new###

		bckgd_fluors=[]
		for frame in range(zstk.shape[2]):
			if self.weights==None:
				self.bckgd_weights(np.copy(zstk[:,:,frame]), zsum, preview=False)
			fluor=np.average(zstk[:,:,frame], weights=self.weights)
			bckgd_fluors.append(fluor)
		bckgd_fluors=np.array(bckgd_fluors)

		for frame in range(zstk.shape[2]):
			sbd = zstk[:,:,frame]-bckgd_fluors[frame]*np.ones((zstk[:,:,frame].shape)) #subtracted img
			clip_im = np.clip(sbd, 0 , sbd)
			self.My_Zstack[:,:,frame] = clip_im



''' #adj_zstk=[]
for frame in range(zstk.shape[2]):
sbd = zstk[:,:,frame]-bckgd_fluors[frame]*np.ones((zstk[:,:,frame].shape)) #subtracted img
clip_im = np.clip(sbd, 0 , sbd)
#adj_zstk.append(clip_im)
self.My_Zstack[:,:,frame] = clip_im '''




####################################
#END OF CLASS Source.#
####################################

class Contours(Source):

	def __init__(self,filename):
		super(Contours, self).__init__(filename)

		#clear when new instance:
		self.mean_values = []
		self.all_contours = None
		self.filtered_contours = None
		self.original = None
		self.canny_edges = None
		self.pulse_frequencies = None


	def contours(self, minVal, maxVal, canny=False, array=None):
		#A contour is a curve joining all the continuous pts 
		#(along the boundary). findContours returns a list of boundrary pts.
		if array==None:
			if self.My_Covar==None:
				self.covar_z()
			array = self.My_Covar

		array *= 255/array.max() 
		copy = array 
		original = array
		array = np.array(array,np.uint8)		

		if canny:
			if self.canny_edges == None:
				self.canny_edge_fn(array=array, minVal=minVal, maxVal=maxVal)
			thresh1 = self.canny_edges

		if not canny:
			#binarization
			thresh1 = cv2.adaptiveThreshold(array,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,17,0)

			# erode/dilate
			kernel = np.ones((1,1),np.uint8)
			opening = cv2.morphologyEx(thresh1,cv2.MORPH_OPEN,kernel, iterations = 2)
			sure_bg = cv2.dilate(opening,kernel,iterations=2)

		# find contours (source img thresh1 mod'd by fxn)
		contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
			
		self.original = copy
		self.all_contours = contours

	def draw_cnts(self, minVal, maxVal, canny=False, array=None):
		#select if canny^
		if self.all_contours == None:
			self.filter_cnts(canny=canny, array=array, minVal=minVal, maxVal=maxVal)
		all_= np.copy(self.filtered_contours)
		copy_= np.copy(self.original)
		img = cv2.drawContours(copy_, all_, -1, (255,255,255), thickness=1) # -1 filled, 1  boundaries.
		display(copy_)


	def filter_cnts(self,  minVal, maxVal, min_area=7, max_area=300, canny=False, array=None): #Changed!!
		if self.all_contours == None:
			self.contours(canny=canny, array=array, minVal=minVal, maxVal=maxVal)

		all_cnts = np.copy(self.all_contours)
		original = np.copy(self.original)

		filt_cnt=[cnt for ind, cnt in enumerate(all_cnts) if max_area>=cv2.contourArea(cnt, oriented=True)>=min_area] #changed oriented=True bc I was having double counts of some contours.
		self.filtered_contours = filt_cnt


	def contour_means(self,  minVal, maxVal, _3D=False, original=None, canny=False, array=None):
		#mean intensity of contours

		''' Note:
		cv2.drawContours(image, contours, contourIdx, color[,...
		#...thickness[, lineType[, hierarchy[, maxLevel[, offset]]]]])->None
		'''

		if self.filtered_contours == None:
			self.filter_cnts(canny=canny, array=array, minVal=minVal, maxVal=maxVal)

		res = list(self.filtered_contours)

		if not(_3D):
			original = np.copy(self.original)
		mean_vals = []
		for i in range(len(res)):
			msk = np.zeros(original.shape,np.uint8) #zero mtx
			cv2.drawContours(msk, res, i, 255, -1)
			mean_ = cv2.mean(original,mask = msk)[0]
			mean_vals.append((i, mean_))
		mean_vals=np.array(mean_vals)

		self.mean_values = mean_vals
		return mean_vals

	def canny_edge_fn(self, array=None, minVal=70, maxVal=175):
		#For Covar: minVal=70, maxVal=175
		#For (255-My_Diff): minVal=125, maxVal=235
		if array == None:
			self.covar_z()
			array = self.My_Covar

		array *= 255/array.max() 
		array = np.array(array,np.uint8)
		edges = cv2.Canny(array,minVal,maxVal,apertureSize = 3)

		self.canny_edges = edges

		#plot
		plt.subplot(121),plt.imshow(array,cmap = 'gray')
		plt.title('Original Image'), plt.xticks([]), plt.yticks([])
		plt.subplot(122),plt.imshow(edges,cmap = 'gray')
		plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
		plt.show()

	###
	###
	###

	def tb(self, use_avg=False, minVal=0, maxVal=160, ratio=3, kernel_size=3):
		#trackbar to pick canny min/max

		if not use_avg:
			if self.My_Covar == None:
					self.covar_z()

			class local:		
				array = self.My_Covar

		if use_avg:
			class local:		
				array = self.My_Avg

		
		def cannythresh(minVal):
			array = local.array
			array *= 255/array.max() 
			array = np.array(array,np.uint8)
			det_edges = cv2.Canny(array,minVal,minVal*ratio,apertureSize = kernel_size)
			dst = cv2.bitwise_and(array, array, mask = det_edges)
			print(minVal, minVal*ratio)
			cv2.imshow('Image', dst)

		while True:
			cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
			minVal = 0
			cv2.createTrackbar('Min Threshold:', 'Image', minVal, maxVal, cannythresh)
			cannythresh(0)
			if cv2.waitKey(1) & 0xFF == ord(' '): #Press space to exit image.
				break
		cv2.destroyAllWindows()


####################################
#END OF CLASS Contours.#
####################################

class Analyze(Contours, Source):
	#simplifying video calcs.

	def __init__(self, filename):
		super(Analyze, self).__init__(filename)
		#clear when new instance
		self.all_cm=None
		self.stdd=None
		self.avg_peak_widths=[]
		self.all_cell_indeces=[]
		self.timed_pulse_interval = []
		self.My_Avg_Fluor = []
		self.blinky_contours = []
		self.avg_blinky = []
		self.avg_nonblinky = []
		self.normed_trace = []

	def big_cm(self, minVal, maxVal, array=None, canny=False):
		#contour means over whole zstack.
		if self.My_Zstack==None:
			self.zstack()
		z=np.copy(self.My_Zstack)
		big_ar=[]
		for f in range(z.shape[2]):
			print('Loading... image: '+str(f))
			big_ar.append(self.contour_means(_3D=True, original=z[:,:,f], canny=canny, array=array, minVal=minVal, maxVal=maxVal))
		# Append class var with all contour means. 
		big_ar = np.array(big_ar)
		self.all_cm = big_ar
		return big_ar


########################################################################
############		 ppeaks Below					       #############
########################################################################

	#create a baseline from troughs in data. 
	#Asymmetric Least Squares Smoothing
	#http://zanran_storage.s3.amazonaws.com/www.science.uva.nl/ContentPages/443199618.pdf
	def baseline_als(self, y, lam, p, niter=10):
		# http://stackoverflow.com/questions/29156532/python-baseline-correction-library# 
		#'fine tune by hand'
		# recommended vals: 0.001 \leq p \leq 0.1; 10^2 \leq lam \leq 10^9
		# niter = iterations 
		L = len(y)
		D = csc_matrix(np.diff(np.eye(L), 2))
		w = np.ones(L)
		for i in xrange(niter):
			W = spdiags(w, 0, L, L)
			Z = W+lam*D.dot(D.transpose())
			z = spsolve(Z, w*y)
			w = p*(y>z)+(1-p)*(y<z)
		return z


	def ppeaks(self, minVal, maxVal, ind=1, mph=1, mpd=1, mph2=-95, mpd2=1, thresh=0, 
		array=None, fps=40, calc_widths=False, minWidth=4, maxWidth=50, avg_fluor=True, 
		show=False, show_vl=False, deltax=25, y_axmin=0, y_axmax=2.2, stddev=None,lam=100, p=.001, niter=6, normed=False):
		''' Peak Detection of Contour Mean Intensities.
		http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
		mph : detect peaks that are greater than minimum peak height.
    	mph2: same, just for valleys (and negative!)
    	mpd : detect peaks that are at least separated by minimum peak distance.
    	mpd2 : same, but valleys
    	threshold : detect peaks (valleys) that are greater (smaller) than `threshold`
        			in relation to their immediate neighbors.

		Seems to work: (see iPython for other population params.)
		for i in range(length):
    		q.ppeaks(ind=i, mph=85, mpd=15,thresh=0.15, minVal=70, maxVal=175)
    	Normed divides each bit of data by its baseline.
		'''
		if self.all_cm == None or self.filtered_contours == None:
			self.big_cm(canny=True, array=array, minVal=minVal, maxVal=maxVal)
		if stddev==None:
			c_m = self.all_cm
			data = c_m[:,ind,1]
		if stddev!=None: #this means look at nonblinky cells (stddev will be low, coded earlier.)
			if self.stdd==None:
				self.create_std(stddev=stddev, avg_fluor=avg_fluor)
			ar = self.stdd
			data = ar[ind,:]

		window = signal.general_gaussian(4, p=0.5, sig=100)
		filtered = signal.fftconvolve(window, data)
		filtered = (np.average(data)/np.average(filtered))*filtered
		filtered = filtered[1:-2] #truncates bad boundaries from fftconvolve
		original_filtered = np.copy(filtered)

		bline=self.baseline_als(filtered, lam=lam, p=p, niter=niter)
		if normed:
			filtered=filtered/bline
		peaks = detect_peaks(x=filtered, mpd=mpd, mph=mph, threshold=thresh, edge='rising', 
									show=True, y_axmin=0, y_axmax=y_axmax)

		#collect the normed traces of all cells. 
		self.normed_trace.append(filtered)

		y_peaks = []
		for ind in peaks:
			y_peaks.append(filtered[ind])
				
		if calc_widths==True:
			self.p_width(peaks=peaks, bline=bline, filtered=filtered, minWidth=minWidth, maxWidth=maxWidth, cell_ind=ind, show=show, data=data, deltax=deltax, normed=normed)
		
		frames = np.arange(0,len(filtered))
		if show_vl==True:
			plt.plot(frames, original_filtered, 'm', frames, bline,'b--') #filtered-->original_filtered
			plt.show()

########################################################################
############		ppeaks Above					       #############
########################################################################

	def p_width(self, peaks, bline, filtered, minWidth, maxWidth, cell_ind, data, show, deltax=25, normed=False):
		if len(peaks)>1:
			tmean_interpeak_dist, fmid = self.mean_int_peak(peaks=peaks, fps=36)
			#deltax = fmid/2.0 #half distance of peak frequency in num. of frames.

			def bline_fn(*args):
				if normed:
					return 1
				return bline[args]

			for i in range(len(peaks)):
				num_frames = len(filtered)
				left_bound = (peaks[i]-deltax)//1
				right_bound = (peaks[i]+deltax)//1

				#Making sure we are in bounds of frames.
				if left_bound<0:
					left_bound=0
				if right_bound>=num_frames:
					right_bound=num_frames-1

				half_max = (filtered[peaks[i]]-bline_fn(peaks[i]))/2.0 

				x1 = np.arange(left_bound, right_bound+1)
				y1 = (np.ones(len(filtered))*(bline_fn(peaks[i])+half_max))[left_bound:right_bound+1] 
				y2 = filtered[left_bound:right_bound+1]

				if show==True:
					plt.plot(x1, y1, '-', x1, y2, '--', x1, bline_fn(x1), '.', peaks[i], filtered[peaks[i]], 'r+', mew=2, ms=8)
					plt.show()

				s = splrep(x1, y2 - (bline_fn(peaks[i])+half_max)) 
				
				roots = sproot(s)
				all_big_widths = []
				cell_indeces = []

				#find smallest interval in roots which... 
				#contains the peak index.
				if len(roots)>1:
					#separate indeces left, right of peak. calc closest.
					more_than_peak=[x for x in it.ifilter(lambda x: x if x>peaks[i] else 0, roots)]
					less_than_peak=[x for x in it.ifilter(lambda x: x if x<peaks[i] else 0, roots)]
					if len(more_than_peak)>0 and len(less_than_peak)>0:
						big_width = min(more_than_peak)-max(less_than_peak)
						print('Width: '+str(big_width))
						all_big_widths.append(big_width)
					else:
						print('No widths detected.')

				else:
					print('None or too few roots detected.')

			if all_big_widths!=[]:
				self.avg_peak_widths.append(np.mean(all_big_widths))
				self.all_cell_indeces.append(cell_ind)

				self.timed_pulse_interval.append(tmean_interpeak_dist)
				print('Mean interpeak duration: '+str(tmean_interpeak_dist)+' seconds')

				#record the contours we use.
				self.blinky_contours.append(data)

	def average_peak_widths_fn(self, fps=40):
		return np.ndarray.flatten(np.array(self.avg_peak_widths)/float(fps))

	def total_mean_fluor(self, cells, blinky=True, smooth=True, window=2, ymax=250, ymin=100):
		#cells is a contour list: self.blinky_contours or self.stdd
		#Shows total mean fluorescence over time of a pop. of cells (blinky or not)
		
		time_fluor_mean = np.mean(cells, axis=0)
		frame = np.arange(time_fluor_mean.shape[0])

		if blinky:
			self.avg_blinky=time_fluor_mean
		if not blinky:
			self.avg_nonblinky=time_fluor_mean

		if smooth:
			# interpolation (scipy fxn). 
			def moving_average(y, window):
				"""Moving average of 'y' with window size 'window'."""
				return lfilter(np.ones(window)/window, 1, y)

			time_fluor_mean2 = moving_average(time_fluor_mean, window=window)
			plt.plot(frame, time_fluor_mean2)
			plt.ylim(ymax=ymax, ymin=ymin)
		
		if not smooth:
			plt.plot(frame, time_fluor_mean)
		
		plt.show()

	def create_std(self, stddev=9, avg_fluor=True):
		c_m = self.all_cm
		stdd_ar = []
		avg_fluor_ar = []
		for i in range(len(self.filtered_contours)):
			if np.std(c_m[:,i,1])<=stddev:
				stdd_ar.append(c_m[:,i,1])
				avg_fluor_ar.append(np.mean(c_m[:,i,1]))
		stdd_ar=np.array(stdd_ar)
		self.stdd=stdd_ar
		self.My_Avg_Fluor=avg_fluor_ar

	def mean_int_peak(self, peaks, fps=4):
		#frames = self.My_Zstack.shape[2]
		if len(peaks)>1:
			distances_bw_peaks,i=[],0
			while i<len(peaks)-1:
				dist=(peaks[i+1]-peaks[i])
				distances_bw_peaks.append(dist)
				i+=1
			mean_dist_bw_peaks = np.mean(distances_bw_peaks) 
			return mean_dist_bw_peaks/float(fps), mean_dist_bw_peaks


	def plot_cm(self, minVal, maxVal, y_axmin=0, y_axmax=255, array=None, smooth=False, peaks=False, mind=1, ind=1, canny=False):
		#Create Plot of Contour Mean Intensities.
		if self.all_cm == None or self.filtered_contours == None:
			self.big_cm(canny=canny, minVal=minVal, maxVal=maxVal, array=array)

		c_m = self.all_cm

		if peaks and not smooth:

			data = c_m[:,ind,1]
			window = signal.general_gaussian(4, p=0.5, sig=100)
			filtered = signal.fftconvolve(window, data)
			filtered = (np.average(data)/np.average(filtered))*filtered
		 		
			detect_peaks(filtered, show=True, mpd = mind, y_axmin=y_axmin, y_axmax=y_axmax)
			#Source: http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

		if smooth:
			xnew=np.linspace(0,len(c_m[:,:,1]),len(c_m[:,:,1]))
			y1=c_m[:,ind,1]#for now must be specified
			f=interp1d(xnew,y1,kind='cubic')(xnew)
			plt.plot(f)
			plt.ylabel('Mean Intensity')
			plt.show()

		if not smooth and not peaks:
			plt.plot(c_m[:,:,1]) 
			plt.ylabel('Mean Intensity')
			plt.show()

		#np.array(big_ar)[:,##CNT_VAL##,1] returns cm vals of CNT_VAL-th contour!
		#np.array(big_ar)[:,##CNT_VAL##,0] returns ind vals of CNT_VAL-th contour!


####################################
#END OF CLASS Analyze.#
####################################

class Meta(Analyze, Contours, Source):

	def __init__(self, filename):
		super(Meta, self).__init__(filename)
	def reset(self):
		dic = vars(self)
		for i in dic.keys():
			dic[i] = []


####################################
#END OF CLASS Meta.#
####################################

# Double Plot
def display(array):
	fig = plt.figure()
	a=fig.add_subplot(1,2,1)
	imgplot = plt.imshow(array, cmap = cm.Greys_r)
	a=fig.add_subplot(1,2,2)
	imgplot = plt.imshow(array, cmap = 'spectral')
	plt.show()

#OpenCV viewer
def display2(array):
	while(True):
		array *= 255/array.max()
		cv2.imshow('Image',np.array(array,dtype=np.uint8))
		if cv2.waitKey(1) & 0xFF == ord(' '): #Press space to exit image.
			break
	cv2.destroyAllWindows()

#Grayscale
def display3(array):
	plt.imshow(array, cmap = cm.Greys_r)
	plt.show()

#Colorful Viewer
def color(array):
	plt.imshow(array)
	plt.show()

#Embeds video into iPython notebook.
def video(fname, mimetype):
    """Load the video in the file `fname`, with given mimetype, and display as HTML5 video.
    """
    from IPython.display import HTML
    video_encoded = open(fname, "rb").read().encode("base64")
    video_tag = '<video controls alt="test" src="data:video/{0};base64,{1}">'.format(mimetype, video_encoded)
    return HTML(data=video_tag)

# Saves image. 
def save_img(fname_string,img):
	cv2.imwrite(fname_string,img)

def saveall(stack):
	for i in range(0,len(stack[0,0,:])):
		cv2.imwrite('masked'+str(i)+'.png',stack[:,:,i])

def mask_z(zstack,mask_img):
	#gets rid of background
	mtx = np.empty((zstack.shape[0], zstack.shape[1]-1, zstack.shape[2]))
	for a in range(0,zstack.shape[2]):
		b=zstack[:,0:zstack.shape[1]-1,a]
		b*=255/b.max()
		b=np.array(b,np.uint8)
		new = cv2.bitwise_and(b,mask)
		mtx[:,:,a]=new
	return mtx

#End of future.py#
########## ########### ########## ########### ########## ########### ########## ###########

#https://gist.github.com/dmeliza/3251476
from matplotlib.offsetbox import AnchoredOffsetbox
class AnchoredScaleBar(AnchoredOffsetbox):
    def __init__(self, transform, sizex=0, sizey=0, labelx=None, labely=None, loc=4,
                 pad=0.1, borderpad=0.1, sep=10, prop=None, **kwargs):
        """
        Draw a horizontal and/or vertical  bar with the size in data coordinate
        of the give axes. A label will be drawn underneath (center-aligned).

        - transform : the coordinate frame (typically axes.transData)
        - sizex,sizey : width of x,y bar, in data units. 0 to omit
        - labelx,labely : labels for x,y bars; None to omit
        - loc : position in containing axes
        - pad, borderpad : padding, in fraction of the legend font size (or prop)
        - sep : separation between labels and bars in points.
        - **kwargs : additional arguments passed to base class constructor
        """
        from matplotlib.patches import Rectangle
        from matplotlib.offsetbox import AuxTransformBox, VPacker, HPacker, TextArea, DrawingArea
        bars = AuxTransformBox(transform)
        if sizex:
            bars.add_artist(Rectangle((0,0), sizex, 0, fc="none"))
        if sizey:
            bars.add_artist(Rectangle((0,0), 0, sizey, fc="none", linewidth=4.0,)) #set linewidth here!
 
        if sizex and labelx:
            bars = VPacker(children=[bars, TextArea(labelx, minimumdescent=False)],
                           align="center", pad=0, sep=sep)
        if sizey and labely:
            bars = HPacker(children=[TextArea(labely), bars],
                            align="center", pad=0, sep=sep)
 
        AnchoredOffsetbox.__init__(self, loc, pad=pad, borderpad=borderpad,
                                   child=bars, prop=prop, frameon=False, **kwargs)
 
def add_scalebar(ax, matchx=True, matchy=True, hidex=True, hidey=True, **kwargs):
    """ Add scalebars to axes

    Adds a set of scale bars to *ax*, matching the size to the ticks of the plot
    and optionally hiding the x and y axes

    - ax : the axis to attach ticks to
    - matchx,matchy : if True, set size of scale bars to spacing between ticks
                    if False, size should be set using sizex and sizey params
    - hidex,hidey : if True, hide x-axis and y-axis of parent
    - **kwargs : additional arguments passed to AnchoredScaleBars

    Returns created scalebar object
    """
    def f(axis):
        l = axis.get_majorticklocs()
        return len(l)>1 and (l[1] - l[0])
    
    if matchx:
        kwargs['sizex'] = f(ax.xaxis)
        kwargs['labelx'] = str(kwargs['sizex'])
    if matchy:
        kwargs['sizey'] = f(ax.yaxis)
        kwargs['labely'] = str(kwargs['sizey'])
        
    sb = AnchoredScaleBar(ax.transData, **kwargs)
    ax.add_artist(sb)
 
    if hidex : ax.xaxis.set_visible(False)
    if hidey : ax.yaxis.set_visible(False)
 
    return sb
