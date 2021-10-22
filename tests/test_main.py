import unittest
import cv2 as cv2
import warnings
import numpy as np

frame = cv2.imread("images/tc6oq6g.jpg")
frame = cv2.resize(frame, (0, 0), fx=5, fy=5)
kernal = np.ones((5, 5), "uint8")
hsvframe = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

red_lower = np.array([0, 150, 150], np.uint8)
red_upper = np.array([10, 255, 255], np.uint8)
red_mask = cv2.inRange(hsvframe, red_lower, red_upper)
red_mask = cv2.dilate(red_mask, kernal)
res_red = cv2.bitwise_and(frame, frame, mask=red_mask)

green_lower = np.array([35, 80, 90], np.uint8)
green_upper = np.array([80, 255, 255], np.uint8)
green_mask = cv2.inRange(hsvframe, green_lower, green_upper)
green_mask = cv2.dilate(green_mask, kernal)
res_green = cv2.bitwise_and(frame, frame, mask=green_mask)

blue_lower = np.array([100, 100, 100], np.uint8)
blue_upper = np.array([130, 255, 255], np.uint8)
blue_mask = cv2.inRange(hsvframe, blue_lower, blue_upper)
blue_mask = cv2.dilate(blue_mask, kernal)
res_blue = cv2.bitwise_and(frame, frame, mask=blue_mask)

yellow_lower = np.array([20, 100, 100], np.uint8)
yellow_upper = np.array([40, 255, 255], np.uint8)
yellow_mask = cv2.inRange(hsvframe, yellow_lower, yellow_upper)
yellow_mask = cv2.dilate(yellow_mask, kernal)
res_yellow = cv2.bitwise_and(frame, frame, mask=yellow_mask)

black_lower = np.array([0, 0, 0], np.uint8)
black_upper = np.array([180, 255, 30], np.uint8)
black_mask = cv2.inRange(hsvframe, black_lower, black_upper)
black_mask = cv2.dilate(black_mask, kernal)
res_black = cv2.bitwise_and(frame, frame, mask=black_mask)

white_lower = np.array([0, 0, 200], np.uint8)
white_upper = np.array([255, 255, 255], np.uint8)
white_mask = cv2.inRange(hsvframe, white_lower, white_upper)
white_mask = cv2.dilate(white_mask, kernal)
res_white = cv2.bitwise_and(frame, frame, mask=white_mask)


countours_green = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
countours_red = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
countours_blue = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
countours_yellow = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
countours_white = cv2.findContours(white_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

class TestDf(unittest.TestCase):

	@classmethod
	def setUpClass(cls):
		print("-" * 50)

	@classmethod
	def tearDownClass(cls):
		print("-" * 50)

	def setUp(self):
		# ignore ResourceWarning
		warnings.simplefilter('ignore', category=ResourceWarning)

	def tearDown(self):
		pass

	def test_red_triangle(self):
		number = 0
		countours = sorted(countours_red, key=cv2.contourArea, reverse=True)[:1]
		for c in countours:
			area = cv2.contourArea(c)
			# test area is float
			self.assertIsInstance(area, float)
			number += 1
		# test number of red triangle is 1
		self.assertEqual(number, 1)

	def test_blue_umbrella(self):
		number = 0
		countours = sorted(countours_blue, key=cv2.contourArea, reverse=True)[:1]
		for c in countours:
			area = cv2.contourArea(c)
			# test area is float
			self.assertIsInstance(area, float)
			number += 1
		# test number of blue umbrella is 1
		self.assertEqual(number, 1)

	def test_green_circle(self):
		number = 0
		countours = sorted(countours_green, key=cv2.contourArea, reverse=True)[:2]
		for c in countours:
			area = cv2.contourArea(c)
			x1, y1, w1, h1 = cv2.boundingRect(c)
			if area > 800 and 1500 < y1 < 2500 and abs(w1 - h1) < 50:
				# test area is float
				self.assertIsInstance(area, float)
				number += 1
		# test number of green circle is 1
		self.assertEqual(number, 1)

	def test_yellow_star(self):
		number = 0
		countours = sorted(countours_yellow, key=cv2.contourArea, reverse=True)[:1]
		for c in countours:
			area = cv2.contourArea(c)
			self.assertIsInstance(area, float)
			number += 1
		self.assertEqual(number, 1)

	def test_white_triangle(self):
		number = 0
		countours = sorted(countours_white, key=cv2.contourArea, reverse=True)[:100]
		for c in countours:
			area = cv2.contourArea(c)
			x1, y1, w1, h1 = cv2.boundingRect(c)
			if 700 < area < 1800 and 2000 < y1 < 2200:
				# test area is float
				self.assertIsInstance(area, float)
				number += 1
		# test number of white triangle is 5
		self.assertEqual(number, 5)


if __name__ == "__main__":
	#run all test
	unittest.main(warnings='ignore')
