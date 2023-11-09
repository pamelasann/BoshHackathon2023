import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from scipy.stats import linregress
from scipy.interpolate import interp1d
import numpy as np
import cv2 as cv

class ROISelector:
    def __init__(self, image, callback):
        self.image = image
        self.callback = callback
        self.roi = None

    def select_callback(self, click, release):
        x1, y1 = click.xdata, click.ydata
        x2, y2 = release.xdata, release.ydata
        self.roi = self.image[int(y1):int(y2), int(x1):int(x2)]
        print(f"x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")
    
    def key_callback(self, event):
        if event.key == "enter" and self.roi is not None:
            self.callback(self.roi)
            return True
        return False

def sigmoid(x, a, b, l, s):
    return a+b*(1/(1+np.power(np.e, -l*(x+s))))

def handle_roi(img):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    height, width = img.shape
    edge = np.zeros(height)
    # a,b,c = params
    line_y = []
    line_x = []
    transects = None
    for idx, row in enumerate(img[::2]):
        i = idx*2
        if max(row) - min(row) < 100:
            # print(f"row {i} discarded, contrast too low!")
            continue
        y_smooth = gaussian_filter(row, sigma=1)
        gradient = np.gradient(y_smooth)
        edge_px = np.argmax(gradient)
        initial_params = [np.min(row), np.max(row), 1.0, -edge_px]
        # print(initial_params)
        try:
            params, cov = curve_fit(sigmoid, range(width), row, p0=initial_params)
            if cov[2][2] > 6e-2:
                # print(f"row {i} discarded, fit failed! [l covariance too high]")
                continue
        except:
            # print(f"row {i} discarded, fit failed!")
            continue
        # print(params)
        inf_pt = -params[3]
        # ax2.plot(range(width)-inf_pt, (sigmoid(range(width), *params)-params[0])/params[1], 'r-')
        edge_data = np.array([range(width)-inf_pt, (sigmoid(range(width), *params)-params[0])/params[1]])
        if transects is None:
            transects = edge_data
        else:
            transects = np.append(transects, edge_data, axis=1)
        ax1.plot(inf_pt, i, 'bx')
        line_x.append(inf_pt)
        line_y.append(i)
        # print(f"row {i} edge at {inf_pt}")
    # ax2.plot(range(width), gradient, 'r-')
    if len(line_x) == 0 or len(line_y) == 0:
        print("No edge found!")
        return -1

    m, b, r2, _, _ = linregress(line_y, line_x)

    transects = np.sort(transects, axis=1)
    [esf_x, esf_y] = np.compress(
        np.logical_and([transects[0] >= -10], [transects[0] <= 10])[0],
        transects,
        axis=1
    )
    esf_p, esf_cov = curve_fit(sigmoid, esf_x, esf_y, p0=[np.min(esf_y), np.max(esf_y), 1.0, 0])


    ax2.plot(esf_x, esf_y, 'b+')
    ax2.plot(esf_x, sigmoid(esf_x, *esf_p), 'm-')
    x_aux = np.linspace(np.min(esf_x), np.max(esf_x), 100)
    lsf_y = np.gradient(sigmoid(x_aux, *esf_p))
    ax2.plot(x_aux, lsf_y, 'g-')

    samp_freq = lsf_y.size/(np.max(x_aux)-np.min(x_aux))
    n = lsf_y.size

    lsf_y = np.append(
        np.append(np.zeros((20*n)), lsf_y),
        np.zeros((20*n))
    )

    lsf_y /= np.sum(lsf_y)
    mtf = np.fft.rfft(lsf_y)
    mtf_freq = np.linspace(0, 0.5*samp_freq, num=mtf.size)
    freq_against_mtf = interp1d(np.abs(mtf), mtf_freq, kind='linear')
    print(f"MTF50: {freq_against_mtf(0.5)}")
    mtf = mtf[mtf_freq <= 0.6]
    mtf_freq = mtf_freq[mtf_freq <= 0.6]
    ax3.plot(mtf_freq, np.abs(mtf), 'r-')

    # ax2.plot(esf_x, np.gradient(sigmoid(esf_x, *esf_p)), 'r-')

    ax1.plot(m*np.arange(height)+b, np.arange(height), 'g-')
    ax1.imshow(img, cmap='gray')
    # fig.show()
    return freq_against_mtf(0.5)

def detectROIs(imageStr):
    image = cv.imread(imageStr, cv.IMREAD_GRAYSCALE)
    thresh = cv.threshold(image, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
    contours, hierarchy = cv.findContours(thresh, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted([
        contour for idx, contour in enumerate(contours)
        if hierarchy[0][idx][3] != -1
    ], key=cv.contourArea, reverse=True)
    if len(contours) == 0:
        print("no contours")
        return None

    epsilon = 0.04 * cv.arcLength(contours[0], True)
    verts = cv.approxPolyDP(contours[0], epsilon, True)
    if len(verts) != 4:
        print("not a rectangle", "verts:", len(verts))
        return None
    verts = sorted(verts, key=lambda x: x[0][0])
    lines = [
        [v[0] for v in sorted(verts[:2], key=lambda x: x[0][1])],
        [v[0] for v in sorted(verts[2:], key=lambda x: x[0][1])]
    ]
    rois = []
    for line in lines:
        x_center = (line[0][0] + line[1][0])//2
        x_padding, y_padding = 30, 30
        rois.append(
            image[line[0][1]+y_padding:line[1][1]-y_padding,line[0][0]-x_padding:line[1][0]+x_padding]
        )
    return rois

def calculate_mtf50(img_path):
    rois = detectROIs(img_path)
    if rois is None:
        print("no rois")
        return None
    mtf = handle_roi(rois[0])


    return mtf > 0.15

if __name__ == '__main__':
    ImageFile = "imgs/19.png"
    # image = cv.imread(ImageFile, cv.IMREAD_GRAYSCALE)
    # _, thresh = cv.threshold(image, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    rois = detectROIs(ImageFile)
    print(handle_roi(rois[0]))
    # if rois is None:
    #     print("no rois")
    #     exit()
    # for roi in rois:
    #     mtf = handle_roi(roi)
    #     print(mtf)
    # plt.imshow(rois[1], cmap='gray')
    # plt.show()
    # gaussian blur
    # image = cv.GaussianBlur(image, (3,3), 0)

    # handler = ROISelector(image, handle_roi)
    
    # fig, ax = plt.subplots()
    # plt.imshow(image, cmap='gray')
    # selector = RectangleSelector(ax, 
    #                              handler.select_callback,
    #                              useblit=True,
    #                              button=[1,3],
    #                              minspanx=5,
    #                              minspany=5,
    #                              spancoords='pixels',
    #                              interactive=True)
    # # mtf = Mtf(ImageFile)
    # plt.connect('key_press_event', handler.key_callback)
    # plt.show()