import cv2
import glob
import os
import ntpath
from IPython import embed
#os.system("mkdir resizedImages")
#os.system("mkdir resizedImages/pos")
#os.system("mkdir resizedImages/neg")

for i, filename in enumerate(glob.glob("pecans/standard/neg/*")):
    print i
    try:
        img = cv2.imread(filename, 1)
        img = cv2.resize(img, (512,512))
        cv2.imwrite("resizedImages/512x512/neg/" + ntpath.basename(filename), img )
    except:
        print "Skipping"
        continue

  
'''  
for i, filename in enumerate(glob.glob("pecans/standard/pos/*")):
    print i
    img = cv2.imread(filename, 1)
    img = cv2.resize(img, (96,96))
    cv2.imwrite("resizedImages/pos/" + ntpath.basename(filename), img )
    
'''
