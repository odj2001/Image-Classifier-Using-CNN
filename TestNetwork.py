import torch
import ImageLoader
import os
import ConvolutionalNetwork
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt



ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PATH = "./net.pth"

model = ConvolutionalNetwork.ConvolutionalNetwork()
model.load_state_dict(torch.load(PATH))
model.eval()

predictedLabel = []
actualLabel = []



batch_size = 1

testData = ImageLoader.loadImages(ROOT_DIR, 'img_list_test.npy')

#Splits the data into batches of size 1
testLoader = torch.utils.data.DataLoader(testData, batch_size=batch_size,
                                         shuffle=False)
with torch.no_grad():
    for i, batch in enumerate(testLoader):
        images = batch["imNorm"]
        labels = batch["label"]

        outputs = model(images)
        
        _, predicted = torch.max(outputs, 1)

        predictedLabel.append(predicted[0])
        actualLabel.append((torch.flatten(labels))[0])
        
        
        
        
        

confusionMatrix = confusion_matrix(actualLabel, predictedLabel)
    
accuracy = confusionMatrix.diagonal()/confusionMatrix.sum(axis=1)
#Prints the accuracy for each of the classses
print("Overall Accuracy: ", sum(accuracy) / len(accuracy))

print("Airplanes Accuracy: ", accuracy[0])
print("Car Accuracy: ", accuracy[1])
print("Dog Accuracy: ", accuracy[2])
print("Faces Accuracy: ", accuracy[3])
print("Keyboard Accuracy: ", accuracy[4])        

target_names = ["Airplanes", "Car", "Dog", "Faces", "Keyboard"]


#Defines the plot and saves it to in the Assignment 2 file
df_cm = pd.DataFrame(confusionMatrix, target_names, target_names)
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
plt.xlabel('Prediction', fontsize=16)
plt.ylabel('Actual', fontsize=16)
plt.savefig('Confusion_Matrix.png',bbox_inches='tight')

        
        
        
        
        
        
        

