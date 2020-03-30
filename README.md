# Project2 - FaceSwap

### Authors: 
1. Saket Seshadri Gudimetla Hanumath ([saketsgh@terpmail.umd.edu](mailto:saketsgh@terpmail.umd.edu), UID: 116332293) 
2. Chayan Kumar Patodi ([ckp1804@terpmail.umd.edu](mailto:ckp1804@terpmail.umd.edu), UID: 116327428)

### Things to Download:

PRNet Model : [https://drive.google.com/file/d/1UoE-XuW1SDLUjZmJPkIZ1MLxvQFgmTFH/view](https://drive.google.com/file/d/1UoE-XuW1SDLUjZmJPkIZ1MLxvQFgmTFH/view) 
Download this model and put in in the path  "/Code/prnet/Data/net-data/"

DLib Model : http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
Download this model and put it in the path "Code/dlib_model/

### How To Run:
To run the program, type the following:-"  python2.7 Wrapper.py  "
The usage of argument parser is given below:

usage: Wrapper.py [-h] [--video VIDEO] [--target TARGET] [--method METHOD] [--mode MODE] [--isDlib ISDLIB]

optional arguments:
  -h, --help       show this help message and exit
  --video VIDEO    Provide Video Name and extension with path here
  --target TARGET  Provide Image to be swapped in the image.
  --method METHOD  Provide Method of image transformation: delaunay(deln), Thin Plate Spline(tps), Position Map Regression Network (prnet)
  --mode MODE      If swapping 1 image in a video, use 1, If swapping 2 faces in a single video, use 2
  --isDlib ISDLIB  True if dlib should be used prediction of facial landmarks. False for using PrNet for the same.

Default arguments are given in the code itself. 

Note: Make sure that all the videos and images, you want to run the code on, are present in the Data and TestSet2_P2 folder present in the current directory.


