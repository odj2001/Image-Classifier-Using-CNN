import torch
import torch.nn as nn
import ImageLoader
import os
import ConvolutionalNetwork

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

trainData = ImageLoader.loadImages(ROOT_DIR, 'img_list_train.npy')

model = ConvolutionalNetwork.ConvolutionalNetwork()
learning_rate = 0.00001

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(trainData)
epochs = 20
batch_size = 16
batches = len(trainData) // batch_size
#Splits the data into batches of size 16

dataLoader = torch.utils.data.DataLoader(dataset=trainData,
                                         batch_size=batch_size,
                                         shuffle=True)

for epoch in range(epochs):
    print("Training Epoch:", epoch)
    for i, batch in enumerate(dataLoader):
        images = batch["imNorm"]
        labels = batch["label"]
        # Run the forward propagate

        outputs = model(images)

        loss = criterion(outputs, labels.flatten().long())

        # Backpropagate Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print('Finished Training')

PATH = "./net.pth"
torch.save(model.state_dict(), PATH)