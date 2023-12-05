import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch
import os

from filters import Filters

xmin, xmax, ymin, ymax = 1200, 2500, 1900, 3200

path_to_image = '/content/drive/MyDrive/IMG_0038.JPG'
path_to_results = '/content/drive/MyDrive/results/'

img=Filters(path_to_image, [xmin, xmax, ymin, ymax])

pass
# # Detect circles using HoughCircles function
# circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, dp=1, minDist=100, param1=23, param2=23, minRadius=50,
#                            maxRadius=80)
# circles = np.uint16(np.around(circles))[0, :, :]
#
# N, M = 2, 2
# fig, axs = plt.subplots(N, M)
# for i, ax in zip(range(N * M), axs.ravel()):
#     x, y, r = circles[i, 0], circles[i, 1], circles[i, 2]
#     ax.imshow(gray[y - r:y + r, x - r:x + r], cmap='gray')
#
# import h5py
# import torch.optim as optim
# from torch.autograd import Variable
# from torch.utils.tensorboard import SummaryWriter
# from sklearn.metrics import f1_score, precision_score, recall_score
#
# import random
# import time
# import pickle
#
# torch.manual_seed(0)
# random.seed(0)
# np.random.seed(0)
# torch.backends.cudnn.deterministic = True
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# print('device_name, ', device)
#
# data = CustomDataset(path)
# train_set, val_set = torch.utils.data.random_split(data, [int(0.8 * len(data)), len(data) - int(0.8 * len(data))])
#
# batch_size = 5
# train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
# test_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size)
#
# model = CNN()
# model = model.to(device)
# print('model on cuda', next(model.parameters()).is_cuda)
# criterion = nn.L1Loss()
# # optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
# optimizer = optim.Adam(model.parameters(), lr=1e-6)
#
# print('model train...')
#
# loss_train_epoch = []
# loss_test_epoch = []
# predicted_labels = {}
# results_labels = {}
#
# # train_log = SummaryWriter('/runs/train/')
# # test_log = SummaryWriter('runs/test/')
#
# for epoch in range(100):
#
#     model.train()
#     start = time.time()
#     train_snr_mean = []
#     batch_loss = []
#     for i, data in enumerate(train_loader):
#         inputs, labels = data
#         inputs = inputs.to(device)
#         labels = labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(inputs.float()[:, np.newaxis, :, :])
#         # outputs=torch.Tensor([1 if i>0.5  else 0 for i in outputs]).to(device)
#         labels = labels.float()
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         batch_loss.append(loss.item())
#         train_snr_mean.append(np.mean([l.item() for l in labels]))
#         if i % 50 == 0:
#             print('epoch', epoch, ',', 'batch', i, ',', 'train loss', loss.item())
#
#     loss_train_epoch.append(sum(batch_loss) / len(batch_loss))
#     # train_log.add_scalar('loss', sum(batch_loss) / len(batch_loss), epoch)
#     print('epoch', epoch, ',', 'loss', sum(batch_loss) / len(batch_loss))
#     print('train_snr_mean', sum(train_snr_mean) / len(train_snr_mean))
#     print('time', time.time() - start)
#
#     model.eval()
#     batch_loss = []
#     predicted_outputs_test = []
#     labels_test = []
#     val_snr_mean = []
#     for i, data in enumerate(test_loader):
#         inputs, labels = data
#         inputs = inputs.to(device)
#         labels = labels.to(device)
#         outputs = model(inputs.float()[:, np.newaxis, :, :])
#         # outputs=torch.Tensor([1 if i>0.5  else 0 for i in outputs]).to(device)
#         labels = labels.float()
#         loss = criterion(outputs, labels)
#         batch_loss.append(loss.item())
#         predicted_outputs_test.extend([i for i in outputs.detach().cpu().numpy()])
#         labels_test.extend([i for i in labels.detach().cpu().numpy()])
#         val_snr_mean.append(np.mean([l.item() for l in labels]))
#
#     predicted_labels.update({epoch: predicted_outputs_test})
#     results_labels.update({epoch: labels_test})
#
#     loss_test_epoch.append(sum(batch_loss) / len(batch_loss))
#     # test_log.add_scalar('loss', sum(batch_loss) / len(batch_loss), epoch)
#     print('test loss', sum(batch_loss) / len(batch_loss))
#     print('val_snr_mean', sum(val_snr_mean) / len(val_snr_mean))
#
#     # if epoch == 0:
#     #     torch.save(model.state_dict(), 'Zarubezneft_model_update_snr_2_15_sn_0.4_1.0_rot.pth')
#     #     print('new best model found in epoch', epoch)
#     # elif loss_test_epoch[epoch - 1] > loss_test_epoch[epoch]:
#     #     torch.save(model.state_dict(), 'Zarubezneft_model_update_snr_2_15_sn_0.4_1.0_rot.pth')
#     #     print('new best model found in epoch', epoch)
#     # if epoch >= 15 and all([loss_test_epoch[epoch] >= loss_test_epoch[j] for j in range(epoch-6, epoch)]):
#     #     break
#
# # train_log.flush()
# # test_log.flush()
# # train_log.close()
# # test_log.close()
#
# # with open('model_update_test_predicted_ZarubezNeft_snr_2_15_sn_0.4_1.0_rot.pkl', 'wb') as f:
# #     pickle.dump(predicted_labels, f)
#
# # with open('model_update_test_labels_ZarubezNeft_snr_2_15_sn_0.4_1.0_rot.pkl', 'wb') as f:
# #     pickle.dump(results_labels, f)
#
# # np.save('model_update_loss_train_ZarubezNeft_snr_2_15_sn_0.4_1.0_rot.npy', np.array(loss_train_epoch))
# # np.save('model_update_loss_test_ZarubezNeft_snr_2_15_sn_0.4_1.0_rot.npy', np.array(loss_test_epoch))
#
# # torch.save(model.state_dict(), 'model_01_11.pth')
#
# plt.plot(loss_test_epoch, marker='*', color='k')
#
# # INFERENCE DATA CREATE
# import cv2
# import numpy as np
# from google.colab.patches import cv2_imshow
# import matplotlib.pyplot as plt
#
# # Load the image
# names_images = ['IMG_0036.JPG']
#
# i = 0
# for name_image in names_images:
#     img = cv2.imread('/content/drive/MyDrive/elecon/' + name_image, cv2.IMREAD_COLOR)  # 30, 9
#
#     # Convert to grayscale.
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     # Apply Gaussian blur to reduce noise
#     img_blur = cv2.medianBlur(img, 5)
#     canny = cv2.Canny(img_blur, 0, 100)
#     img_blur = img_blur[1200:2500, 1900:3200]
#     canny = canny[1200:2500, 1900:3200]
#
#     # Detect circles using HoughCircles function
#     circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, dp=1, minDist=100, param1=23, param2=23, minRadius=50,
#                                maxRadius=80)
#
#     # Draw detected circles on the original image
#     if circles is not None:
#         circles = np.round(circles[0, :]).astype("int")
#         for (x, y, r) in circles:
#             cv2.imwrite('/content/drive/MyDrive/inference/' + str(i) + '.png',
#                         gray[1200:2500, 1900:3200][y - r:y + r, x - r:x + r])
#             i = i + 1
#
# # INFERENCE
# import os
#
# path_to_inference_data = '/content/drive/MyDrive/inference/'
# # inference_data = CustomDataset(path)
# # batch_size = 2
# # inference_loader = torch.utils.data.DataLoader(inference_data, batch_size=batch_size)
#
# model.eval()
# labels_test = []
# predicted_outputs_test = []
# # for i, data in enumerate(test_loader):
# for name_image in os.listdir(path_to_inference_data):
#     image = cv2.resize(cv2.imread(path_to_inference_data + name_image, cv2.IMREAD_GRAYSCALE), (150, 150))
#     output = model(torch.from_numpy(image).to(device).float()[np.newaxis, :, :])
#     print(name_image, round(output.item()))
#     # print(model(torch.from_numpy(image).to(device).float()[np.newaxis,:,:]))
#     # inputs, labels = data
#     # inputs = inputs.to(device)
#     # labels = labels.to(device)
#     # outputs = model(inputs.float()[:,np.newaxis,:,:])[:, 0]
#     # predicted_outputs_test.extend([i for i in outputs.detach().cpu().numpy()])
#
# fig = plt.figure(figsize=(5, 5))
# ax = fig.subplots(1, 6)
#
# image = cv2.resize(cv2.imread(path_to_inference_data + '0.png', cv2.IMREAD_GRAYSCALE), (150, 150))
# ax[0].imshow(image, cmap='gray')
# ax[0].set_title('predicted 1, 0.png')
#
# image = cv2.resize(cv2.imread(path_to_inference_data + '1.png', cv2.IMREAD_GRAYSCALE), (150, 150))
# ax[1].imshow(image, cmap='gray')
# ax[1].set_title('predicted 0, 1.png')
#
# image = cv2.resize(cv2.imread(path_to_inference_data + '10.png', cv2.IMREAD_GRAYSCALE), (150, 150))
# ax[2].imshow(image, cmap='gray')
# ax[2].set_title('predicted 1, 10.png')
#
# image = cv2.resize(cv2.imread(path_to_inference_data + '26.png', cv2.IMREAD_GRAYSCALE), (150, 150))
# ax[3].imshow(image, cmap='gray')
# ax[3].set_title('predicted 1, 26.png')
#
# image = cv2.resize(cv2.imread(path_to_inference_data + '29.png', cv2.IMREAD_GRAYSCALE), (150, 150))
# ax[4].imshow(image, cmap='gray')
# ax[4].set_title('predicted 0, 29.png')
#
# image = cv2.resize(cv2.imread(path_to_inference_data + '31.png', cv2.IMREAD_GRAYSCALE), (150, 150))
# ax[5].imshow(image, cmap='gray')
# ax[5].set_title('predicted 0, 31.png')
#
# fig.subplots_adjust(right=4)
#
# # plt.imshow(img_blur,cmap = 'gray') #[1200:2500,1900:3200]
# ticks = [i for i in range(0, img_blur.shape[1], 40)]
# labels = [i for i in range(0, img_blur.shape[1], 40)]
# plt.xticks(ticks, labels, rotation=90, fontsize=8)
# ticks = [i for i in range(0, img_blur.shape[0], 40)]
# labels = [i for i in range(0, img_blur.shape[0], 40)]
# plt.yticks(ticks, labels, fontsize=8)
# plt.imshow(img_blur)
#
# import cv2
# import numpy as np
# from google.colab.patches import cv2_imshow
# import matplotlib.pyplot as plt
#
# # Load the image
# names_images = ['IMG_0037.JPG', 'IMG_0038.JPG', 'IMG_0039.JPG']
#
# i = 0
# for name_image in names_images:
#     img = cv2.imread('/content/drive/MyDrive/elecon/' + name_image, cv2.IMREAD_COLOR)  # 30, 9
#
#     # Convert to grayscale.
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     # Apply Gaussian blur to reduce noise
#     img_blur = cv2.medianBlur(img, 5)
#     canny = cv2.Canny(img_blur, 0, 100)
#     img_blur = img_blur[1200:2500, 1900:3200]
#     canny = canny[1200:2500, 1900:3200]
#
#     # Detect circles using HoughCircles function
#     circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, dp=1, minDist=100, param1=23, param2=23, minRadius=50,
#                                maxRadius=80)
#
#     # Draw detected circles on the original image
#     if circles is not None:
#         circles = np.round(circles[0, :]).astype("int")
#         for (x, y, r) in circles:
#             cv2.imwrite('/content/drive/MyDrive/results/' + str(i) + '.png',
#                         gray[1200:2500, 1900:3200][y - r:y + r, x - r:x + r])
#             i = i + 1
#
# # AUGMENTATION FOR 0 CLASS
# import cv2
# import numpy as np
# from google.colab.patches import cv2_imshow
# import matplotlib.pyplot as plt
# import os
#
# names_images = os.listdir('/content/drive/MyDrive/results/0/')
# coeff_blur = np.arange(1, 23, 2)
#
# i = 0
# for name_image in names_images:
#     img = cv2.imread('/content/drive/MyDrive/results/0/' + name_image, cv2.IMREAD_COLOR)
#     # Convert to grayscale.
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     for c_blurr in coeff_blur:
#         img_blur = cv2.medianBlur(gray, c_blurr)
#         cv2.imwrite('/content/drive/MyDrive/results/0_0/' + str(i) + '.png', img_blur)
#         i = i + 1
#
# import cv2
# import numpy as np
# from google.colab.patches import cv2_imshow
# import matplotlib.pyplot as plt
#
# # Load the image
# img = cv2.imread('/content/drive/MyDrive/elecon/IMG_0036.JPG', cv2.IMREAD_COLOR)
#
# # Convert to grayscale.
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # Apply Gaussian blur to reduce noise
# # img_blur = cv2.GaussianBlur(gray,(5,5),1)
# img_blur = cv2.medianBlur(img, 5)
# canny = cv2.Canny(img_blur, 0, 100)
# img_blur = img_blur[1200:2500, 1900:3200]
# canny = canny[1200:2500, 1900:3200]
#
# # plt.imshow(img_blur,cmap = 'gray') #[1200:2500,1900:3200]
# # ticks= [i for i in range(0,img_blur.shape[1],20)]
# # labels= [i for i in range(0,img_blur.shape[1],20)]
# # plt.xticks(ticks,labels,rotation=-90,fontsize=8)
# # ticks= [i for i in range(0,img_blur.shape[0],20)]
# # labels= [i for i in range(0,img_blur.shape[0],20)]
# # plt.yticks(ticks,labels,fontsize=8)
# # img_blur=cv2.line(img_blur, (200,700), (300,700), (0, 255, 0), 9)
# # cv2_imshow(img_blur)
#
# # Detect circles using HoughCircles function
# circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, dp=1, minDist=100, param1=23, param2=23, minRadius=50,
#                            maxRadius=80)
# # circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, dp=1, minDist=100, param1=3, param2=3, minRadius=50, maxRadius=80)
#
# # Draw detected circles on the original image
# if circles is not None:
#     circles = np.round(circles[0, :]).astype("int")
#     for (x, y, r) in circles:
#         cv2.circle(img[1200:2500, 1900:3200], (x, y), r, (0, 255, 0), 2)
#
# # Display the final image with detected circles
# # cv2.imshow('Detected Circles', img_blur[1200:1960,1900:2500])
# cv2_imshow(img[1200:2500, 1900:3200])
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# from google.colab.patches import cv2_imshow
#
# !curl - o
# logo.png
# https: // colab.research.google.com / img / colab_favicon_256px.png
# import cv2
#
# img = cv2.imread('logo.png', cv2.IMREAD_UNCHANGED)
# cv2_imshow(img)
