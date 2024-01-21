!pip install roboflow

from roboflow import Roboflow

#Paste private API
#Contact me in Telegramm https://t.me/N9021010

# creating Roboflow object and connecting to API
rf = Roboflow(api_key="paste your API_KEY")

# upload API parametrs
project = rf.workspace("mipt23").project("fox_pic")
dataset = project.version(2).download("yolov5")
model = project.version(2).model

#paste your pic with a fox
img_url = " "

# get json request from server
predictions = model.predict(img_url, hosted=True, confidence=70, overlap=30).json()

print(predictions)
