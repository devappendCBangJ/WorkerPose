import glob
import json
import cv2

path = "C://Users//dohyu//Desktop//hackathon//F01_05//F01Image-001"  # images이 저장된 파일 경로(폴더)
path1 = "C://Users//dohyu//Desktop//hackathon//F01_05//F01Image-001" # json이 저장된 파일 경로(폴더)

images = glob.glob(path+'//*.jpg')
jsons = glob.glob(path1+'//*.json')

# images = images.sort()
# jsons = jsons.sort()

images_number = 100 # 이미지 몇 번째 불러올지

img = cv2.imread(str(images[images_number]))

with open(str(jsons[images_number]), 'r') as f:
    worker = json.load(f)
    print(type(worker), worker)
    print(worker["annotations"][0]["keypoints"])
    for item in worker["annotations"][0]["keypoints"]:
        cv2.circle(img, (int(item[0]), int(item[1])), 5, (0,255,0), -1)

# print(str(images[0]))
# cv2.imwrite('C:\\Users\\dohyu\\Desktop\\worker_s.png',img)
# 사진이 너무 크니까 바탕화면에 저장하고 쓰세요~
cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
