import os
from tqdm import tqdm
flist=[]
for root, subFolder, files in tqdm(os.walk("/home/JJ_Group/yinsz/data/imagenet/train")):
    for item in files:
        if item.endswith(".jpeg") or item.endswith(".JPEG"):
            fileNamePath = str(os.path.join(root,item))
            flist.append(fileNamePath)
print(len(flist))
