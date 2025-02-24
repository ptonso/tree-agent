import cv2
import numpy as np

def main():

    def update(val):
        scale = cv2.getTrackbarPos('Scale', 'Image')
        img = np.ones((300, 300, 3), dtype=np.uint8) * scale  # Modify brightness
        cv2.imshow('Image', img)

    cv2.namedWindow('Image')
    cv2.createTrackbar('Scale', 'Image', 0, 255, update)

    update(0)  # Show initial image
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()