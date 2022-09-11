import torch
import torch.nn as nn
import torch.functional as F
import matplotlib.pyplot as plt
import numpy as np
from ai4mars_dataset import AI4MarsDataset
from unet_50 import UNet
import pickle
from sklearn.cross_validation import train_test_split


def main(num_epochs=5, batch_size=4, image_size = 256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    dataset = np.load('low1.npz')
    labelset = np.load('high1.npz')
    # B = np.load('final2.npz')
    df = dataset['velocity']
    lf = labelset['velocity']

    x_train, x_test, y_train, y_test = train_test_split(df, lf, test_size=0.2)

    print(f"No. of training examples: {x_train.shape[0]}")
    print(f"No. of testing examples: {x_test.shape[0]}")
    print(f"No. of training labels: {y_train.shape[0]}")
    print(f"No. of testing labels: {y_test.shape[0]}")

    model = UNet(in_channels=1, n_classes=1)
    record = [[], [], []]


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    model.to(device)
    best_model_test_acc = 0
    
    for epoch in range(num_epochs):  # loop over the dataset multiple timesi
        
        running_loss = 0.0
        train_total, test_total = 0, 0
        train_correct, test_correct = 0, 0
        model.train()
        
        for data in trainloader:
            # get the inputs; data is a list of [inputs, labels]
            images, depths, labels = data
            images=images.reshape(batch_size,1,image_size,image_size)
            depths=depths.reshape(batch_size,1,image_size,image_size)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            
            outputs = model(images.to(device))
            
            labels = labels.reshape(batch_size,image_size,image_size).long()
            labels[labels==255]=4
            print(outputs.shape)
            print(labels.shape)
            
            loss = criterion(outputs, labels.to(device))
            print('loss_in')
            loss.backward()
            print('loss_out')
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            seg_acc = ((labels.cpu() == torch.argmax(outputs, axis=1).cpu()).sum() / torch.numel(labels.cpu())).item()
            print('batch_acc:',seg_acc)
            _, train_predict = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (train_predict == labels.to(device)).sum().item()
            print('loss:',loss.item())
    
        with torch.no_grad():
            model.eval()
            for data in testloader:
                images, depths, labels = data
                images=images.reshape(batch_size,1,image_size,image_size)
                depths=depths.reshape(batch_size,1,image_size,image_size)
                outputs = model(images.to(device))
                _, test_predict = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (test_predict == labels.to(device)).sum().item()
        
        train_acc = train_correct/train_total
        test_acc = test_correct/test_total
        record[0].append(train_acc)
        record[1].append(test_acc)
        record[2].append(running_loss/len(trainloader))
        print('Epoch %d| Train loss: %.4f| Train Acc: %.3f| Test Acc: %.3f'%(
            epoch+1, running_loss/len(trainloader), train_acc, test_acc))
        if test_acc>best_model_test_acc:
            best_model_test_acc=test_acc
            
            torch.save(model.state_dict(), PATH)
            with open("./record.pkl", "wb") as fp:   # Unpickling
                pickle.dump(record, fp)

    print('Finished Training')
  

    return record


if __name__ == "__main__":
    main()