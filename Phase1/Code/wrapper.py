import cv2
import numpy as np
import copy
import random

from warp_triangulation import SwapFaceWithImg, SwapTwoFacesInVid, showImage

def main():
    FilePathTestSet = '../../Data/TestSet_P2/'
    FilePathCustomSet = '../../Data/'
    Test1 = cv2.VideoCapture(FilePathTestSet+'Test1.mp4')
    Test2 = cv2.VideoCapture(FilePathTestSet+'Test2.mp4')
    Test3 = cv2.VideoCapture(FilePathTestSet+'Test3.mp4')
    CustomSet = cv2.VideoCapture(FilePathCustomSet+'Video9.mp4')
    # Define the codec and create VideoWriter object
    #     fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #     out = cv2.VideoWriter('FaceSwap.avi',fourcc, 24.0, (640,480), True)
    SwapFaceWithImg_ = False
    rambo = cv2.imread('../../Data/TestSet_P2/Rambo.jpg')
    scarlett = cv2.imread('../../Data/TestSet_P2/Scarlett.jpg')
    i = 0
    frameCount = 0

    
    if(SwapFaceWithImg_):
        while(Test3.isOpened()):
            print("frame - {}".format(frameCount))
            ret, img = Test3.read()
            if ret==True:
                swapImg, OneFaceDetected = SwapFaceWithImg(img, scarlett)
                # if two faces not found then drop the current frame 
                if(not OneFaceDetected):
                    frameCount += 1
                    continue
                else:
    #                     showImage(swapImg)
                    cv2.imwrite("results/Test3/img"+str(i)+".png", swapImg)
                    #out.write(swapImg)
                    i+=1
                    frameCount+=1
            else:
                break
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
        Test3.release()
    #     out.release()
        cv2.destroyAllWindows()
        print("done")

    else:
        while(Test2.isOpened()):
            print("frame - {}".format(frameCount))
            ret, img = Test2.read()
    #             img = cv2.flip(img, 0)
            if ret==True:
                swapImg, TwoFacesDetected = SwapTwoFacesInVid(img)
                # if two faces not found then drop the current frame 
                if(not TwoFacesDetected):
                    frameCount += 1
                    continue
                else:
                    showImage(swapImg)
                    cv2.imwrite("results/Test2/img"+str(i)+".png", swapImg)
                    i+=1
                    frameCount += 1
            else:
                break
        Test2.release()
        cv2.destroyAllWindows()
        print("done")

if __name__ == '__main__':
	main()