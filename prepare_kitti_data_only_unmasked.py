#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import glob
import os
import tqdm
import cv2

# sample
# /data/raw_dataset/widerface/WIDER_train/images/0--Parade/0_Parade_Parade_0_605.jpg| 18,77,47,116,1 128,92,146,116,1 282,270,324,325,1 483,50,521,102,1 544,78,580,125,1 724,43,760,87$

images_train = sorted(glob.glob('/data/nks/k/MaskedFace/data/kitti_out/train/images/*'))
labels_train = sorted(glob.glob('/data/nks/k/MaskedFace/data/kitti_out/train/labels/*'))
images_val = sorted(glob.glob('/data/nks/k/MaskedFace/data/kitti_out/test/images/*'))
labels_val = sorted(glob.glob('/data/nks/k/MaskedFace/data/kitti_out/test/labels/*'))

assert len(images_train) == len(labels_train)
assert len(images_val) == len(labels_val)


def parse_kitti_file(images, labels):
    img_path = []
    faces = []
    for i in tqdm.tqdm(range(len(labels))):
        img_path.append(images[i])

        with open(labels[i], 'r') as fr:
            lines = fr.readlines()
        lines = [x.split() for x in lines]

        face_loc = []
        for k in lines:
            if k[0] == 'Mask':
                continue
            else:
                loc = [int(float(k[4])), int(float(k[5])), int(float(k[6])), int(float(k[7]))]
                loc[0] = max(0,loc[0])
                loc[1] = max(0,loc[1])
                loc[2] = max(0,loc[2])
                loc[3] = max(0,loc[3])

                h,w,_ = cv2.imread(images[i]).shape

                loc[0] = min(w, loc[0])
                loc[1] = min(h, loc[1])
                loc[2] = min(w, loc[2])
                loc[3] = min(h, loc[3])


                if loc[0] < loc[2] and loc[1] < loc[3]:
                    face_loc += [loc]
        faces += [face_loc]
    return img_path, faces


def kitti_data_file():

    img_paths, bbox = parse_kitti_file(images_train, labels_train)
    fw = open('train.txt', 'w')
    for index in range(len(img_paths)):
        tmp_str = ''
        tmp_str =tmp_str+ img_paths[index]+'|'
        boxes = bbox[index]

        for box in boxes:
            data = ' %d,%d,%d,%d,1'%(box[0], box[1], box[2],  box[3])
            tmp_str=tmp_str+data
        if len(boxes) == 0:
            print(tmp_str)
            tmp_str += ' '
            # continue
        ####err box?
        if box[2] <= 0 or box[3] <= 0:
            pass
        else:
            fw.write(tmp_str + '\n')
    fw.close()

    img_paths, bbox = parse_kitti_file(images_val, labels_val)

    fw = open('val.txt', 'w')
    for index in range(len(img_paths)):

        tmp_str=''
        tmp_str =tmp_str+ img_paths[index]+'|'
        boxes = bbox[index]

        for box in boxes:
            data = ' %d,%d,%d,%d,1'%(box[0], box[1], box[2],  box[3])
            tmp_str=tmp_str+data

        if len(boxes) == 0:
            print(tmp_str)
            tmp_str += ' '

            # continue
        ####err box?
        if box[2] <= 0 or box[3] <= 0:
            pass
        else:
            fw.write(tmp_str + '\n')
    fw.close()


if __name__ == '__main__':
    kitti_data_file()