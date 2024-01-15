import cv2
import numpy as np

src=cv2.imread('src.jpg')
dst=cv2.imread('dst.jpg')

srcs=src.shape
dsts=dst.shape
coptemp=dst.copy()

pts1 = np.float32([[940,96],[1427,395],[455,395],[943,1022]])
pts2 = np.float32([[450,33],[540,300],[362,302],[450,567]])

pts3 = np.float32([[943,395]])

M = cv2.getPerspectiveTransform(pts1,pts2)

players = np.array([[1438.3716,  541.7197, 1463.7478,  602.2346],
        [1421.6936,  408.3843, 1445.0157,  456.7011],
        [1028.5947,  394.3304, 1044.3032,  450.7318],
        [1468.7223,  579.1996, 1498.0638,  635.0728],
        [1373.2493,  356.6744, 1394.0450,  389.9394],
        [ 776.7164,  285.4023,  793.4136,  327.3116],
        [1137.1399,  345.5762, 1160.5884,  391.6691],
        [ 809.9502,  442.8443,  833.5020,  494.9818],
        [1061.7950,  262.3989, 1074.8494,  300.0678],
        [1509.3220,  327.9478, 1545.0444,  355.2290],
        [1231.0175,  267.6180, 1245.1418,  300.0868],
        [1482.0227,  432.9044, 1527.4259,  464.7272],
        [1491.7560,  421.2573, 1528.2427,  462.3718]])

for p in players:
    tl = np.float32([[p[0], p[1]]])
    br = np.float32([[p[2], p[3]]])

    # Apply perspective transformation
    tl_transformed = cv2.perspectiveTransform(tl[None, :, :], M)
    br_transformed = cv2.perspectiveTransform(br[None, :, :], M)

    # Extract transformed coordinates
    x1, y1 = int(tl_transformed[0][0][0]), int(tl_transformed[0][0][1])
    x2, y2 = int(br_transformed[0][0][0]), int(br_transformed[0][0][1])

    # Draw the bounding box on the destination image
    # cv2.rectangle(coptemp, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.circle(coptemp, (x1, y1), 15, (255,0,0),-1)

    # x=p[0]+int(p[2]/2)
    # y=p[1]+p[3]
    # pts3 = np.float32([[x,y]])
    # pts3o=cv2.perspectiveTransform(pts3[None, :, :],M)
    # x1=int(pts3o[0][0][0])
    # y1=int(pts3o[0][0][1])
    # pp=(x1,y1)
    # cv2.circle(coptemp,pp, 15, (255,0,0),-1)

cv2.imshow('sada',coptemp)

cv2.waitKey(0)

cv2.destroyAllWindows()
