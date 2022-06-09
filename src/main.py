import sys
from cv2 import cv2
import numpy as np
import time

# define the image path
path = r"../images/tc6oq6g.jpg"
# define first time for showing image
first_time = time.time()

#OpenCV main loop
while True:
	# Read image
	frame = cv2.imread(path)
	# resize image for more accurate detection
	frame = cv2.resize(frame, (0, 0), fx=5, fy=5)
	# Convert to hsv color space
	hsvframe = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	# kernel for morphological operation
	kernal = np.ones((5, 5), "uint8")

	# define range of red color in HSV
	red_lower = np.array([0, 150, 150], np.uint8)
	red_upper = np.array([10, 255, 255], np.uint8)
	# mask of red color
	red_mask = cv2.inRange(hsvframe, red_lower, red_upper)
	# perform morphological operations
	red_mask = cv2.dilate(red_mask, kernal)
	# bitwise and mask and original image
	res_red = cv2.bitwise_and(frame, frame, mask=red_mask)

	# define range of green color in HSV
	green_lower = np.array([35, 80, 90], np.uint8)
	green_upper = np.array([80, 255, 255], np.uint8)
	# mask for green color
	green_mask = cv2.inRange(hsvframe, green_lower, green_upper)
	# perform morphological operations
	green_mask = cv2.dilate(green_mask, kernal)
	# bitwise and mask and original image
	res_green = cv2.bitwise_and(frame, frame, mask=green_mask)

	# define range of blue color in HSV
	blue_lower = np.array([100, 100, 100], np.uint8)
	blue_upper = np.array([130, 255, 255], np.uint8)
	# mask for blue color
	blue_mask = cv2.inRange(hsvframe, blue_lower, blue_upper)
	# perform morphological operations
	blue_mask = cv2.dilate(blue_mask, kernal)
	# bitwise and mask and original image
	res_blue = cv2.bitwise_and(frame, frame, mask=blue_mask)

	yellow_lower = np.array([20, 100, 100], np.uint8)
	yellow_upper = np.array([40, 255, 255], np.uint8)
	yellow_mask = cv2.inRange(hsvframe, yellow_lower, yellow_upper)
	yellow_mask = cv2.dilate(yellow_mask, kernal)
	res_yellow = cv2.bitwise_and(frame, frame, mask=yellow_mask)

	# define range of white color in HSV
	white_lower = np.array([0, 0, 200], np.uint8)
	white_upper = np.array([255, 255, 255], np.uint8)
	# mask for white color
	white_mask = cv2.inRange(hsvframe, white_lower, white_upper)
	# perform morphological operations
	white_mask = cv2.dilate(white_mask, kernal)
	# bitwise and mask and original image
	res_white = cv2.bitwise_and(frame, frame, mask=white_mask)

	# find appropriate contours for each color
	countours_green = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
	countours_red = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
	countours_blue = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
	countours_yellow = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
	countours_white = cv2.findContours(white_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

	# wait 9 seconds for show red_triangle
	if time.time() - first_time > 9:
		# sort contours by area for more accurate detection
		countours = sorted(countours_red, key=cv2.contourArea, reverse=True)[:1]
		for contour in countours:
			area = cv2.contourArea(contour)
			if area > 900:
				# define bounding box
				x1, y1, w1, h1 = cv2.boundingRect(contour)
				# draw bounding box
				frame_red_bar = cv2.rectangle(
					frame, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 255), 10
				)
				# put shape name on image
				cv2.putText(
					frame,
					"red triangle",
					(x1 + w1, y1 + h1),
					cv2.FONT_HERSHEY_SIMPLEX,
					5,
					(0, 0, 255),
					5,
				)
	# wait 6 seconds for show green_circle
	if time.time() - first_time > 6:
		# sort contours by area for more accurate detection
		countours = sorted(countours_green, key=cv2.contourArea, reverse=True)[:2]
		for contour in countours:
			# define bounding box
			area = cv2.contourArea(contour)
			# define bounding box
			x1, y1, w1, h1 = cv2.boundingRect(contour)
			if area > 800 and 1500 < y1 < 2500 and abs(w1 - h1) < 50:
				# draw bounding box
				frame_green_bar = cv2.rectangle(
					frame, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 10
				)
				# put shape name on image
				cv2.putText(
					frame,
					"green circle",
					(x1 + w1, y1 + h1),
					cv2.FONT_HERSHEY_SIMPLEX,
					5,
					(0, 255, 0),
					5,
				)

	# wait 3 seconds for show blue_umbrella
	if time.time() - first_time > 3:
		# sort contours by area for more accurate detection
		countours = sorted(countours_blue, key=cv2.contourArea, reverse=True)[:1]
		for contour in countours:
			area = cv2.contourArea(contour)
			if area > 800:
				# define bounding box
				x1, y1, w1, h1 = cv2.boundingRect(contour)
				# draw bounding box
				frame_blue_bar = cv2.rectangle(
					frame, (x1, y1), (x1 + w1, y1 + h1), (255, 0, 0), 10
				)
				# put shape name on image
				cv2.putText(
					frame,
					"blue umbrella",
					(x1 + w1, y1 + h1),
					cv2.FONT_HERSHEY_SIMPLEX,
					5,
					(255, 0, 0),
					5,
				)
	# wait 12 seconds for show yellow_star
	if time.time() - first_time > 12:
		# sort contours by area for more accurate detection
		countours = sorted(countours_yellow, key=cv2.contourArea, reverse=True)[:1]
		for contour in countours:
			area = cv2.contourArea(contour)
			if area > 800:
				# define bounding box
				x1, y1, w1, h1 = cv2.boundingRect(contour)
				# draw bounding box
				frame_yellow_bar = cv2.rectangle(
					frame, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 255), 10
				)
				# put shape name on image
				cv2.putText(
					frame,
					"yellow star",
					(x1 + w1, y1 + h1),
					cv2.FONT_HERSHEY_SIMPLEX,
					5,
					(0, 255, 255),
					5,
				)
	# wait 15 seconds for show white_triangle
	if time.time() - first_time > 15:
		# sort contours by area for more accurate detection
		countours = sorted(countours_white, key=cv2.contourArea, reverse=True)[:100]
		for contour in countours:
			area = cv2.contourArea(contour)
			# define bounding box
			x1, y1, w1, h1 = cv2.boundingRect(contour)
			if 700 < area < 1800 and 2000 < y1 < 2200:
				# draw bounding box
				frame_white_bar = cv2.rectangle(
					frame, (x1, y1), (x1 + w1, y1 + h1), (255, 255, 255), 10
				)
				# put shape name on image
				cv2.putText(
					frame,
					"white triangle",
					(x1 + w1, y1 + h1),
					cv2.FONT_HERSHEY_SIMPLEX,
					5,
					(255, 255, 255),
					5,
				)
	# resize image for normal size and show
	frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)
	cv2.imshow("main", frame)
	cv2.setWindowProperty("main", cv2.WND_PROP_TOPMOST, 1)
	# press q to quit
	if cv2.waitKey(1) & 0xFF == ord("q"):
		cv2.destroyAllWindows()
		cv2.waitKey(1)
		sys.exit()

# 1337
