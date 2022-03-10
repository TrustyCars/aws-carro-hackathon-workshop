import cv2
import matplotlib.pyplot as plt

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5): 
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        # cv2 Image
        img = cv2.imread(str(img))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])

def easyocr_detection_save_results(image_file, image, results, dirname):
    """
    save result from easyocr text detection 
    
    image_file : (Path) image file path
    image      : cv2 image 
    bbox       : (np.ndarray) bbox result from easyocr detect
    dirname    : (Path) output dir
    
    """
    for bbox in results[0][0]:
        cv2.rectangle(image, (bbox[0], bbox[2]), (bbox[1], bbox[3]),  (0, 255, 0), 2)
    

    for bbox in results[1][0]:
        (tl, tr, br, bl) = bbox
        tl = (int(tl[0]), int(tl[1]))
        tr = (int(tr[0]), int(tr[1]))
        br = (int(br[0]), int(br[1]))
        bl = (int(bl[0]), int(bl[1]))



        cv2.rectangle(image, tl, br, (0, 255, 0), 2)

    new_filename = dirname.joinpath('res_' + image_file.stem + image_file.suffix)

    cv2.imwrite(str(new_filename), image)
    
    
def paddleocr_detection_save_results(image_file, image, results, dirname):
    """
    save result from easyocr text detection 
    
    image_file : (Path) image file path
    image      : cv2 image 
    bbox       : (np.ndarray) bbox result from easyocr detect
    dirname    : (Path) output dir
    
    """
    

    for bbox in results[0]:
        
        cv2.rectangle(image, tuple(bbox[0].astype("int")), tuple(bbox[2].astype("int")), (0, 255, 0), 2)

    new_filename = dirname.joinpath('res_' + image_file.stem + image_file.suffix)

    cv2.imwrite(str(new_filename), image)
    
def get_text_only_easyocr(results):
    text = [i[1] for i in results]
    return ' '.join(text)

def get_text_only_paddleocr(results):
    text = [i[1][0] for i in results]
    return ' '.join(text)

def get_list_of_corners_cor(bbox):
    """

    :param bbox:
    :return:
    """
    x1, y1, x2, y2, x3, y3, x4, y4 = [int(tmp) for tmp in bbox.flatten()]
    return x1, y1, x2, y2, x3, y3, x4, y4


def get_corners(bbox):
    """
    Return the corners coordinates of bbox
    :param bbox:
    :return:
    """
    x1, y1, x2, y2, x3, y3, x4, y4 = bbox
    top_left_x = min([x1,x2,x3,x4])
    top_left_y = min([y1,y2,y3,y4])
    bot_right_x = max([x1,x2,x3,x4])
    bot_right_y = max([y1,y2,y3,y4])

    return top_left_x, top_left_y, bot_right_x, bot_right_y


def crop_bbox(image, bbox):
    """
    Crop a bounding box
    :param image:
    :param bbox:
    :return:
    """
    top_left_x, top_left_y, bot_right_x, bot_right_y = get_corners(get_list_of_corners_cor(bbox))

    return image[top_left_y:bot_right_y+1, top_left_x:bot_right_x+1]

def get_width(bbox):
    """

    :param bbox:
    """
    top_left_x, top_left_y, bot_right_x, bot_right_y = get_corners(bbox)

    return bot_right_x - top_left_x