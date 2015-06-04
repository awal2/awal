#Author: Alex Walczak 2015
from __future__ import division, print_function
import time
from PIL import Image, ImageDraw
import numpy as np
import math
import cv2
import pyaudio
#from random import randint
#import pyautogui as PAG

# PIL code created with suggestions from "Managing Your Biological Data with Python" by Via et al.
# and stackoverflow.com/questions/4160175/detect-tap-with-pyaudio-from-live-mic
'''Shows a clock, its background is determined by luminance detected by webcam.
Clicking 's' key toggles size; spacebar exits clock.
Tapping noises also toggle an effect.'''

class Clock(object):

	global TRACKER
	
	def __init__(self):

		self.CONST_COLOR = 'lightgrey'
		self.color = TRACKER.color
		if TRACKER.startSmall:
			self.SIZE = (400,400)
		if not TRACKER.startSmall:
			self.SIZE = (700,700)




		self.CENTER = (int(self.SIZE[0]/2), int(self.SIZE[1]/2))
		self.clock = Image.new('L', self.SIZE, 'grey')
		self.DRAW_CLOCK = ImageDraw.Draw(self.clock)
		self.BOUNDINGBOX = (50, 50, self.SIZE[0]-50, self.SIZE[1]-50)
		incmt = int(self.SIZE[0]/30)
		self.smallBBOX = (self.CENTER[0]-incmt,self.CENTER[0]-incmt,self.CENTER[0]+incmt,self.CENTER[0]+incmt)
		self.DRAW_CLOCK.pieslice(self.BOUNDINGBOX, 0, 360, fill = self.color)#fill='royalblue')
		self.DRAW_CLOCK.arc(self.BOUNDINGBOX, 0, 360, fill = 'black')
		self.RADIUS = (self.SIZE[0]-100)/2


	@staticmethod
	def click(event, x, y, flags, param):
		if event == cv2.EVENT_LBUTTONDOWN:
			TRACKER.count+=1 
	

	def blank(self):
		self.__init__()
	
	
	def getTime(self):
		self.hour = time.localtime().tm_hour
		self.minutes = time.localtime().tm_min
		self.seconds = time.localtime().tm_sec
	
	def drawTriangle(self, start_angle, color, radius, end_pt):
		p1 = Clock.coord(start_angle+90, end_pt, -radius)
		p2 = Clock.coord(start_angle, end_pt, 1.68*radius)
		p3 = Clock.coord(start_angle-90, end_pt, -radius)
		self.DRAW_CLOCK.polygon((p1, p2, p3), fill = color)
		
	def Run(self):
		self.getTime()

		hour_ang, min_ang, sec_ang = Clock.time_to_angle(self.hour, self.minutes, self.seconds)

		self.blank()

		hr_pt = Clock.coord(hour_ang, self.CENTER, self.RADIUS/1.7)
		min_pt = Clock.coord(min_ang, self.CENTER, self.RADIUS/1.18)
		sec_pt = Clock.coord(sec_ang, self.CENTER, self.RADIUS/1.1)

		front_color = self.CONST_COLOR

		for i in range(12):
			i+=1
			ang = 360*(i/12)
			self.drawTriangle(ang, 'darkgrey', self.RADIUS/40, Clock.coord(ang, self.CENTER, self.RADIUS/1.09))

		self.DRAW_CLOCK.line( ( self.CENTER, hr_pt ), width = int(self.SIZE[0]/33), fill=front_color)
		#self.drawTriangle(hour_ang, front_color, int(self.SIZE[0]/25), hr_pt)

		self.DRAW_CLOCK.line( ( self.CENTER, min_pt ), width = int(self.SIZE[0]/56), fill=front_color )
		#self.drawTriangle(min_ang, front_color, int(self.SIZE[0]/35), min_pt)

		self.DRAW_CLOCK.line( ( self.CENTER, sec_pt ), width = int(self.SIZE[0]/181), fill=front_color )
		self.drawTriangle(sec_ang, front_color, int(self.SIZE[0]/199), sec_pt)

		self.DRAW_CLOCK.pieslice(self.smallBBOX, 0, 360, fill = 'grey')
		#self.clock.show()
		return self.clock
	

	def Show(self):
		cv2.namedWindow("Clock", cv2.WINDOW_NORMAL)
		#cv2.setMouseCallback("Clock", Clock.click)

		# display the image and wait for a keypress
		video_capture = cv2.VideoCapture(0)
		AUDIOSTREAM = TRACKER.MICAUDIO.open(format=TRACKER.FORMAT, channels=1, rate=TRACKER.RATE, input=True, frames_per_buffer=TRACKER.CHUNKSIZE)

		while True:

			key = cv2.waitKey(1) & 0xFF              

			# Capture frame-by-frame
			ret, frame = video_capture.read()
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			
			if not TRACKER.GOTLOUD:
				TRACKER.color = int(np.mean(frame))
			if TRACKER.GOTLOUD:
				TRACKER.color = 'white'

			cv2.imshow("Clock", np.array(self.Run()))# [:, :, ::-1])
			
			#Clicking s toggles the size of the window.
			if key == ord('s') or key == ord('S'):
				TRACKER.startSmall = not TRACKER.startSmall
				if TRACKER.startSmall:
					cv2.resizeWindow("Clock", 400, 422) #22px = size of MacOSX window bar
				if not TRACKER.startSmall:
					cv2.resizeWindow("Clock", 700, 722)

			#toggle clock color if tap detected.
			if key == ord(' '):
				break
			try:
				data = AUDIOSTREAM.read(TRACKER.CHUNKSIZE)
				data = Clock.get_rms(data)
			except IOError, e:
				TRACKER.NOISYCOUNT = 1
				#print('IOError ignored :D.')
				a=0 

			#print(data)

			if data > TRACKER.TAPTHRESH:
				# noisy block
				TRACKER.QUIETCOUNT = 0
				TRACKER.NOISYCOUNT += 1


				if TRACKER.NOISYCOUNT > TRACKER.OVERSENS:
					# turn down the sensitivity
					print("It's loud in here! I'll lower sensitivity.")
					TRACKER.TAPTHRESH *= 1.03
					TRACKER.NOISYCOUNT = 0  #reset noisy block count


				elif TRACKER.NOISYCOUNT <= TRACKER.OVERSENS:
					if TRACKER.NOISYCOUNT==1: #don't want flashes. (else see MAXYNOISY...)
						print('woof!')
						TRACKER.GOTLOUD = not TRACKER.GOTLOUD


			elif data <= TRACKER.TAPTHRESH:		
				TRACKER.QUIETCOUNT += 1
				TRACKER.NOISYCOUNT = 0 
				if TRACKER.QUIETCOUNT > TRACKER.UNDERSENS:
					print("It's quiet in here. I'm upping sensitivity.")
					# turn up the sensitivity
					TRACKER.TAPTHRESH *= 0.92
					TRACKER.QUIETCOUNT = 0 #reset noisy block count

			#print(data, TRACKER.TAPTHRESH, TRACKER.QUIETCOUNT,TRACKER.NOISYCOUNT)



		AUDIOSTREAM.stop_stream()
		AUDIOSTREAM.close()
		TRACKER.MICAUDIO.terminate()

		print('* done recording *')
		video_capture.release()
		cv2.destroyAllWindows()




	@staticmethod
	def time_to_angle(hours, minutes, seconds=None):
		hours = hours%12
		hour_pos = ((hours/12)*360-90+30*(minutes/60))%360  #want hr hand to move diff amt based on minutes. 360/12 = 30
		min_pos = ((minutes/60)*360-90)%360
		sec_pos = ((seconds/60)*360-90)%360
		return hour_pos, min_pos, sec_pos

	@staticmethod
	def coord(angle, center, radius):
		angle = ((90-angle)/180)*np.pi
		x = center[0] + math.sin(angle)*radius
		y = center[1] + math.cos(angle)*radius
		return (x,y)

	@staticmethod
	def get_rms( block ):
		#Citation:
		#stackoverflow.com/questions/4160175/detect-tap-with-pyaudio-from-live-mic
		# RMS amplitude is defined as the square root of the 
		# mean over time of the square of the amplitude.
		# so we need to convert this string of bytes into 
		# a string of 16-bit samples...

		# we will get one short out for each 
		# two chars in the string.
		count = len(block)/4
		shorts = np.fromstring(block,dtype=np.int8).flatten()

		# iterate over the block.
		sum_squares = 0.0
		for sample in shorts:
		    # sample is a signed short in +/- 32768. 
		    # normalize it to 1.0
		    n = sample * (1.0/255.0) #normalize wrt 255 = 2^8-1.
		    sum_squares += n*n
		return math.sqrt( sum_squares / count )


class Tracker(object):
	count = 0
	startSmall = True
	color = 0

 #1024 #samples per block. buffer size...
	FORMAT = pyaudio.paInt16 #paInt8
	RATE = 44100 #sample rate

	MICAUDIO = pyaudio.PyAudio()
	RATE = int(MICAUDIO.get_device_info_by_index(0)['defaultSampleRate'])
	print(RATE)
	CHUNKSIZE = int(RATE*0.076)
	print(CHUNKSIZE)
	GOTLOUD = False
	TAPTHRESH = 0.4 #initial thresh value.

	MAXNOISYBLOCKS = 3
	NOISYCOUNT = 0 # If >'x' chunks in a row are noisy, something's wrong.
	QUIETCOUNT = 0 
	OVERSENS = 6 #MAX NOISY CHUNKS IN A ROW
	UNDERSENS = 150 #MAX QUIET CHUNKS IN A ROW

#Make counter.
TRACKER = Tracker()
	
#Create an instance.
My_Clock = Clock()

#Show
My_Clock.Show()
