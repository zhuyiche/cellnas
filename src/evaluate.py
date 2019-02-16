import cv2
import numpy as np
import os
epsilon = 1e-7

def non_max_suppression(img, overlap_thresh=0.3, max_boxes=1200, r=8, prob_thresh=0.8):
    x1s = []
    y1s = []
    x2s = []
    y2s = []
    probs = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] < prob_thresh:
                img[i, j] = 0
            else:
                x1 = max(j - r, 0)
                y1 = max(i - r, 0)
                x2 = min(j + r, img.shape[1] - 1)
                y2 = min(i + r, img.shape[0] - 1)
                x1s.append(x1)
                y1s.append(y1)
                x2s.append(x2)
                y2s.append(y2)
                probs.append(img[i, j])
    x1s = np.array(x1s)
    y1s = np.array(y1s)
    x2s = np.array(x2s)
    y2s = np.array(y2s)
    # print(x1s.shape)
    boxes = np.concatenate((x1s.reshape((x1s.shape[0], 1)), y1s.reshape((y1s.shape[0], 1)),
                            x2s.reshape((x2s.shape[0], 1)), y2s.reshape((y2s.shape[0], 1))), axis=1)
    # print(boxes.shape)
    probs = np.array(probs)
    pick = []
    area = (x2s - x1s) * (y2s - y1s)
    indexes = np.argsort([i for i in probs])

    while len(indexes) > 0:
        last = len(indexes) - 1
        i = indexes[last]
        pick.append(i)

        xx1_int = np.maximum(x1s[i], x1s[indexes[:last]])
        yy1_int = np.maximum(y1s[i], y1s[indexes[:last]])
        xx2_int = np.minimum(x2s[i], x2s[indexes[:last]])
        yy2_int = np.minimum(y2s[i], y2s[indexes[:last]])

        ww_int = np.maximum(0, xx2_int - xx1_int)
        hh_int = np.maximum(0, yy2_int - yy1_int)

        area_int = ww_int * hh_int
        # find the union
        area_union = area[i] + area[indexes[:last]] - area_int

        # compute the ratio of overlap
        overlap = area_int / (area_union + 1e-6)

        indexes = np.delete(indexes, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

        if len(pick) >= max_boxes:
            break
            # return only the bounding boxes that were picked using the integer data type
    boxes = boxes[pick]
    # print(boxes.shape)

    return boxes

"""
def score_single_img(input, img_dir, prob_threshold=None, print_single_result=True):
    input = misc.imresize(input, (500, 500))
    input = input / 255.
    boxes = non_max_suppression(input, prob_thresh=prob_threshold)

    num_of_nuclei = boxes.shape[0]

    mat_path = os.path.join(IMG_DIR, img_dir, img_dir + '_detection.mat')
    #print(mat_path)
    gt = sio.loadmat(mat_path)['detection']
    #outputbase = cv2.imread(os.path.join(IMG_DIR, img_dir, img_dir + '.bmp'))
    if print_single_result:
        print('----------------------------------')
        print('This is {}'.format(img_dir))
        print('detected: {}, ground truth: {}'.format(num_of_nuclei, gt.shape[0]))

    pred = []
    for i in range(boxes.shape[0]):
        x1 = boxes[i, 0]
        y1 = boxes[i, 1]
        x2 = boxes[i, 2]
        y2 = boxes[i, 3]
        cx = int(x1 + (x2 - x1) / 2)
        cy = int(y1 + (y2 - y1) / 2)
        # cv2.rectangle(outputbase,(x1, y1), (x2, y2),(255,0,0), 1)
        #cv2.circle(outputbase, (cx, cy), 3, (255, 255, 0), -1)
        pred.append([cx, cy])
    p, r, f1, tp, aaa, bbb = get_metrics(gt, pred)
    return p, r, f1, tp, aaa, bbb

def eval_single_img(model, img_dir, print_img=True,
                    prob_threshold=None, print_single_result=True):
    image_path = os.path.join(IMG_DIR, img_dir, img_dir+ '.bmp')
    img = misc.imread(image_path)
    img = misc.imresize(img, (256, 256))#, interp='nearest')
    img = img - 128.0
    img = img / 128.0
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    output = model.predict(img)[0]
    output = output[:, :, 1]
    def _predic_crop_image(img, print_img=print_img):
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        output = model.predict(img)[0]
        output = output[:, :, 1]
        return output
    
    #if print_img:
    #    plt.imshow(output)
    #    plt.title(weight_path)
    #    plt.colorbar()
    #    plt.show()
    
    #print(output.shape)

    p, r, f1, tp, gt, pred2 = score_single_img(output, img_dir=img_dir, prob_threshold=prob_threshold,
                                               print_single_result=print_single_result)
    return p, r, f1, tp, gt, pred2
"""

def get_metrics(pred, mask, gt, r=6):
    pred = np.array(pred).astype('int')
    pred_map = np.zeros((256,256),dtype='int')
    for i in range(pred.shape[0]):
        x = pred[i, 0]
        y = pred[i, 1]
        pred_map[y, x] = 1

    result_map = mask * pred_map
    tp = result_map.sum()

    precision = min(tp / (pred.shape[0] + epsilon),1)
    recall = min(tp / (gt + epsilon),1)
    f1_score = 2 * (precision * recall / (precision + recall + epsilon))
    gt_num = gt
    pred_num = pred.shape[0]
    return precision, recall, f1_score, tp, gt_num, pred_num


def single_img_score(output, mask, gt, prob_thresh):
    boxes = non_max_suppression(output, prob_thresh=prob_thresh)
    pred = []
    for i in range(boxes.shape[0]):
        x1 = boxes[i, 0]
        y1 = boxes[i, 1]
        x2 = boxes[i, 2]
        y2 = boxes[i, 3]
        cx = int(x1 + (x2 - x1) / 2)
        cy = int(y1 + (y2 - y1) / 2)
        # cv2.rectangle(outputbase,(x1, y1), (x2, y2),(255,0,0), 1)
        # cv2.circle(input, (cx, cy), 3, (255, 255, 0), -1)
        pred.append([cx, cy])
    p, r, f1, tp, gt_num, pred_num= get_metrics(pred, mask, gt)
    return p, r, f1, tp, gt_num, pred_num


def eval(output, mask, gt, prob_thresh):
    n = mask.shape[0]
    total_p, total_r, total_f1, total_tp = 0, 0, 0, 0
    tp_total_num, gt_total_num, pred_total_num = 0, 0, 0
    for i in range(n):
        curr_output = output[i, 1, :, :]
        #curr_output = curr_output.reshape((curr_output.shape[1], curr_output.shape[2]))
        curr_mask = mask[i, :, :, :]
        curr_gt = gt[i]
        p, r, f1, tp, gt_num, pred_num = single_img_score(output=curr_output, mask=curr_mask,gt=curr_gt, prob_thresh=prob_thresh)
        total_p += p
        total_r += r
        total_f1 += f1
        total_tp += tp

        tp_total_num += tp
        gt_total_num += gt_num
        pred_total_num += pred_num
    precision = tp_total_num / (pred_total_num + epsilon)
    recall = tp_total_num / (gt_total_num + epsilon)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
    return f1_score
    #return precision, recall, f1_score, total_p / n, total_r / n, total_f1 / n

"""
def eval_testset(model, prob_threshold=None, print_img=False, print_single_result=True):
    total_p, total_r, total_f1, total_tp = 0, 0, 0, 0
    tp_total_num, gt_total_num, pred_total_num = 0, 0, 0
    for img_dir in os.listdir(IMG_DIR):
        p, r, f1, tp, gt, pred = eval_single_img(model, img_dir, print_img=print_img,
                                       print_single_result=print_single_result,
                                       prob_threshold=prob_threshold)
        #print('{} p: {}, r: {}, f1: {}, tp: {}'.format(img_dir, p, r, f1, tp))
        total_p += p
        total_r += r
        total_f1 += f1
        total_tp += tp

        tp_total_num += tp
        gt_total_num += gt
        pred_total_num += pred

    precision = tp_total_num/(pred_total_num + epsilon)
    recall = tp_total_num / (gt_total_num + epsilon)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
    #print('Over points, the precision: {}, recall: {}, f1: {}'.format(precision, recall, f1_score))
    #if prob_threshold is not None:
        #print('The nms threshold is {}'.format(prob_threshold))
    #print('Over test set, the average P: {}, R: {}, F1: {}, TP: {}'.format(total_p/20,total_r/20,total_f1/20, total_tp/20))
    return precision, recall, f1_score, total_p/20, total_r/20, total_f1/20, total_tp/20
"""