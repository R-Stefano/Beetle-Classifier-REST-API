import requests, os
path_text="urls.txt"

o=open(path_text, "r")
urls=o.read()
o.close()

urls=urls.split()

newPath="testData/"

os.makedirs("testData", exist_ok=True)

for idx,url in enumerate(urls):
    try:
        img=requests.get(url).content
        if(img[:2]==b'\xff\xd8'):
            print("Image",idx," :",img[:2])
            f=open(newPath + "image_"+str(idx)+".jpg", "wb")
            f.write(img)
            f.close()
    except Exception as e:
        print("\n\nException",e,"\n\n")
        pass