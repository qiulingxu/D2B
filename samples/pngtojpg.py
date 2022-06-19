import glob
from PIL import Image
import os
import shutil

for root, dirs, files in os.walk("."):
    for name in files:
        #if name.endswith((".jpg")):
        #    os.remove(os.path.join(root,name))
        if name.endswith((".png")):
            print(root,name)
            fo = name.split(".")[0]
            img = Image.open(os.path.join(root,name))
            img = img.convert('RGB')
            img.save(os.path.join(root,fo+".jpg"))
            os.remove(os.path.join(root,name))
        try:
            if name.endswith((".jpg")) and int(name.split(".")[0])>105:
                os.remove(os.path.join(root,name))
        except:
            pass
