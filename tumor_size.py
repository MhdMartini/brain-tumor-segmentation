import argparse
import os
import cv2
import numpy as np


class TumorSize:
    def __init__(self, scan_area, seg_width, dir):
        self.vols = {}  # tumors' volumes
        self.scan_area = scan_area
        self.seg_width = seg_width
        self.dir = dir
        self.rows, self.cols = 0, 0

        self.dir = dir
        self.calculate()

    def get_area(self, image):
        # get number of white pixels in an image
        height, width = image.shape
        num = 0
        for i in range(height):
            for j in range(width):
                if image[i, j] == 255:
                    num += 1
        return num

    def fill_boundaries(self, boundary):
        # find centroids and fill contours
        mask = np.copy(boundary)
        pattern = [0, 255, 0, 255, 0]

        for i in range(self.rows):
            pattern = [0, 255, 0, 255, 0]
            pos_255 = []
            for j in range(self.cols):
                val = mask[i, j]
                if val == pattern[0]:
                    pattern.pop(0)
                    if val == 255:
                        pos_255.append(j)
                        continue

                    if not pattern:
                        mask[i, pos_255[0]: pos_255[1]] = 255
                        break
        return mask

    def refine_mask(self, mask, boundary):
        boundary_inv = cv2.bitwise_not(boundary)
        mask = cv2.bitwise_and(mask, boundary_inv)
        return mask

    def calculate(self):
        os.chdir(self.dir)
        if not os.path.exists("output"):
            os.mkdir("output")
        images_names = os.listdir(os.getcwd())
        for N, image_name in enumerate(sorted(images_names)):
            if "." not in image_name:
                continue
            img = cv2.imread(image_name, 0)
            self.rows, self.cols = img.shape

            _, boundary = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY)
            mask = self.fill_boundaries(boundary)
            mask = self.refine_mask(mask, boundary)

            # preprocess image before applying mask
            area = self.get_area(mask)
            size = int((area / np.pi) ** 0.5)
            size -= (size % 2) - 1
            thresh_type = cv2.ADAPTIVE_THRESH_MEAN_C
            image_proc = cv2.adaptiveThreshold(
                img, 255, thresh_type, cv2.THRESH_BINARY, size, 0)
            result = cv2.bitwise_and(mask, image_proc)
            path = os.path.join("output", image_name.replace(
                image_name.split(".")[0], image_name.split(".")[0] + "_tumor"))
            cv2.imwrite(path, result)

            pixels_num = result.shape[0] * result.shape[1]
            area_per_pixel = self.scan_area / pixels_num
            tumor_area_pixel = self.get_area(result)
            tumor_area_msquared = tumor_area_pixel * area_per_pixel
            self.vols[image_name] = tumor_area_msquared * self.seg_width

        total = 0
        for name, vol in self.vols.items():
            vol *= 1e6  # to cm cubed
            vol = round(vol, 3)
            total += vol
            print(f"Tumor Area in {name}\t= {vol}\tcm\u00b3")
        print()
        print(f"Tumor Total Area: {round(total, 3)} cm\u00b3")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Calculated brain tumor volume from outlined MRI images')
    parser.add_argument('--scan_shape', type=float, nargs='+', default=(0.4, 0.4),
                        help='height and width of scan area in meters')
    parser.add_argument('--seg_width', type=float, default=2e-3,
                        help='distance in meters between two consecutive MRI scans')
    parser.add_argument('dir', type=str,
                        help='directory containing MRI scans')
    args = parser.parse_args()

    scan_area = args.scan_shape[0] * args.scan_shape[1]
    TumorSize(scan_area, args.seg_width, args.dir)
