from glob import glob
import yaml

img_list = glob("C://DataSet//images//*.jpg")

from sklearn.model_selection import train_test_split

train_img_list, val_img_list = train_test_split(img_list, test_size=0.2, random_state=2000)
with open('C://DataSet//train.txt', 'w') as f:
    f.write('\n'.join(train_img_list) + '\n')
with open('C://DataSet//val.txt', 'w') as f:
    f.write('\n'.join(val_img_list) + '\n')

with open('C://DataSet//data.yaml', 'r') as d:
    data = yaml.full_load(d)

data['train'] = 'C://DataSet//train.txt'
data['val'] = 'C://DataSet//val.txt'
with open('C://DataSet//data.yaml', 'w') as d:
    yaml.dump(data, d)

print(data)
