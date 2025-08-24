# Red Ball Detection

A computer vision application that detects and tracks red spherical objects in video streams using OpenCV's HSV color filtering and contour analysis.

## Overview

This system processes video frames to identify red balls by converting images to HSV color space, applying dual-range color masks to handle red's wraparound in the hue spectrum, and using morphological operations to reduce noise. The detected object's centroid is calculated using image moments and visualized with contour overlays.

## Requirements
I'm using conda here but on usual python interpreter it will work as well.
```bash
pip install opencv-python numpy
```

## Usage

Place your video file as `rgb_ball_720.mp4` and run:

```python
python main.py
```

The system will process each frame, highlight detected red objects with blue contours, and mark the centroid with a circle and text label. Press 'q' to exit the video stream.

## Configuration

HSV color ranges can be adjusted in the detection parameters to accommodate different lighting conditions and red object variations. The dual-range approach handles red's position at both ends of the hue spectrum (0° and 180°) for robust detection across various red tones.

## Key Components

- **detect_red_ball()**: Performs HSV filtering, contour detection, and morphological noise reduction
- **centroid()**: Calculates object center using moments and overlays visualization markers
- **imshow()**: Utility for scaled image display during development and debugging