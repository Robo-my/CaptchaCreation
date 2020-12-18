import os, random
import cv2, argparse
import numpy as np


backgroundlist = os.listdir('background')


def image_augmentation(img, type2=False):
    # perspective
    w, h, _ = img.shape
    pts1 = np.float32([[0, 0], [0, w], [h, 0], [h, w]])
    # 좌표의 이동점
    begin, end = 30, 90
    pts2 = np.float32([[random.randint(begin, end), random.randint(begin, end)],
                       [random.randint(begin, end), w - random.randint(begin, end)],
                       [h - random.randint(begin, end), random.randint(begin, end)],
                       [h - random.randint(begin, end), w - random.randint(begin, end)]])
    M = cv2.getPerspectiveTransform(pts1, pts2)

    img = cv2.warpPerspective(img, M, (h, w))

    # Brightness
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img = np.array(img, dtype=np.float64)
    random_bright = .4 + np.random.uniform()
    img[:, :, 2] = img[:, :, 2] * random_bright
    img[:, :, 2][img[:, :, 2] > 255] = 255
    img = np.array(img, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

    # Blur
    blur_value = random.randint(0,4) * 2 + 1
    img = cv2.blur(img,(blur_value, blur_value))
    if type2:
        return img[130:280, 180:600, :]
    return img[130:280, 120:660, :]


class ImageGenerator:
    def __init__(self, save_path):
        self.save_path = save_path
        # Plate
        self.plate = cv2.imread("plate.jpg")

        # loading Number
        file_path = "num/"
        file_list = os.listdir(file_path)
        self.Number = list()
        self.number_list = list()
        for file in file_list:
            img_path = os.path.join(file_path, file)

            self.Number.append(img_path)
            self.number_list.append(file)

        # loading Char
        file_path = "char/"
        file_list = os.listdir(file_path)
        self.char_list = list()
        self.Char1= list()
        for file in file_list:
            img_path = os.path.join(file_path, file)
            #img = cv2.imread(img_path)
            self.Char1.append(img_path)
            self.char_list.append(file)
        
    def Type_1(self, num, save=False):
        number = self.Number
        char = self.Char1

        for i, Iter in enumerate(range(num)):
            try:
                Plate = cv2.resize(self.plate, (520, 110))
                b_width ,b_height = 800, 400
                # random_R, random_G, random_B = random.randint(0,255), random.randint(0,255), random.randint(0,255)
                # background = np.zeros(( b_height,b_width, 3), np.uint8)
                # cv2.rectangle(background, (0, 0), (b_width, b_height), (random_R, random_G, random_B), -1)
                rand_int = random.randint(0, len(backgroundlist))
                background = cv2.imread('background/'+backgroundlist[rand_int])

                

                label = ""
                # row -> y , col -> x
                row, col = 13, 35  # row + 83, col + 56
                rand_int = random.randint(0, 25)
                label += self.char_list[rand_int]
                charImg = cv2.imread(char[rand_int]+'/'+str(random.randint(0, 24))+'.jpg')
                h,w,_= charImg.shape
                Plate[row:row + h, col:col + w, :] = charImg
                col += w

                # character 2
                rand_int = random.randint(0, 25)
                label += self.char_list[rand_int]
                charImg = cv2.imread(char[rand_int]+'/'+str(random.randint(0, 24))+'.jpg')
                h,w,_= charImg.shape
                Plate[row:row + h, col:col + w, :] = charImg
                col += w

                # character 3
                rand_int = random.randint(0, 25)
                label += self.char_list[rand_int]
                charImg = cv2.imread(char[rand_int]+'/'+str(random.randint(0, 24))+'.jpg')
                h,w,_= charImg.shape
                Plate[row:row + h, col:col + w, :] = charImg
                col += (w + 36)

                # number 4
                rand_int = random.randint(0, 9)
                label += self.number_list[rand_int]
                numImg = cv2.imread(number[rand_int]+'/'+str(random.randint(0, 24))+'.jpg')
                h,w,_= numImg.shape
                Plate[row:row + h, col:col + w, :] = numImg
                col += w

                # number 5
                rand_int = random.randint(0, 9)
                label += self.number_list[rand_int]
                numImg = cv2.imread(number[rand_int]+'/'+str(random.randint(0, 24))+'.jpg')
                h,w,_= numImg.shape
                Plate[row:row + h, col:col + w, :] = numImg
                col += w

                # number 6
                rand_int = random.randint(0, 9)
                label += self.number_list[rand_int]
                numImg = cv2.imread(number[rand_int]+'/'+str(random.randint(0, 24))+'.jpg')
                h,w,_= numImg.shape
                Plate[row:row + h, col:col + w, :] = numImg
                col += w

                # number 7
                rand_int = random.randint(0, 9)
                label += self.number_list[rand_int]
                numImg = cv2.imread(number[rand_int]+'/'+str(random.randint(0, 24))+'.jpg')
                h,w,_= numImg.shape
                Plate[row:row + h, col:col + w, :] = numImg
                col += w


                s_width, s_height = int((400-110)/2), int((800-520)/2)
                background[s_width:110 + s_width, s_height:520 + s_height, :] = Plate
                background = image_augmentation(background)

                if save:
                    print(self.save_path + label + ".jpg")
                    cv2.imwrite(self.save_path + label + ".jpg", background)
                else:
                    cv2.imshow(label, background)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            except Exception as e:
                print(e)


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--img_dir", help="save image directory",
                    type=str, default="DB/")
parser.add_argument("-n", "--num", help="number of image",
                    type=int)
parser.add_argument("-s", "--save", help="save or imshow",
                    type=bool, default=True)
args = parser.parse_args()


img_dir = args.img_dir
A = ImageGenerator(img_dir)

num_img = args.num
Save = args.save
#%%
A.Type_1(10, save=Save)
print("Type 1 finish")


# A.Type_2(num_img, save=Save)
# print("Type 2 finish")
# A.Type_3(num_img, save=Save)
# print("Type 3 finish")
# A.Type_4(num_img, save=Save)
# print("Type 4 finish")
# A.Type_5(num_img, save=Save)
# print("Type 5 finish")
