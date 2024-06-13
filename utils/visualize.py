import cv2
import os
import torch

# batchsize should set as 1
class painter1():
    def __init__(self, image, data_gt, data_pred):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image = image
        self.data_pred = data_pred
        self.data_gt = data_gt[:15]
        self.data_gt_future = data_gt[15:]
        
        self.viz_line()
    
    def viz_line(self):
        im_path = 'C:\\Users\\user\\Desktop\\' + self.data_gt.sceneId.unique()[0] + '.jpg'
        
        if os.path.exists(self.image):
            im = cv2.imread(self.image)
            gt_points = self.data_gt.iloc[:, 2:4].values.tolist()
            gt_future_points = self.data_gt_future.iloc[:, 2:4].values.tolist()
            
            for g in self.data_pred:
                s_past = None
                for j in g:
                    #print(((torch.Tensor(gt_future_points).to(self.device) - j) ** 2).mean())
                    if ((torch.Tensor(gt_future_points).to(self.device) - j) ** 2).mean() > 10000:
                        print('skip')
                        continue
                    for s in j:
                        if s_past is not None:
                            im = cv2.line(im, (int(s_past[0]), int(s_past[1])), (int(s[0]), int(s[1])), color=(0, 255, 0), thickness=2)
                        s_past = s
            
            g_past = None
            for g in gt_points:
                if g_past is not None:
                    im = cv2.line(im, (int(g_past[0]), int(g_past[1])), (int(g[0]), int(g[1])), color=(255, 255, 0), thickness=2)
                g_past = g
            
            g_past = None
            for g in gt_future_points:
                if g_past is not None:
                    im = cv2.line(im, (int(g_past[0]), int(g_past[1])), (int(g[0]), int(g[1])), color=(0, 0, 255), thickness=2)
                g_past = g
            
            cv2.imwrite(im_path, im)
    
    def viz_bbox(self):
        im_path = 'C:\\Users\\user\\Desktop\\' + self.data_gt.sceneId.unique()[0] + '_bbox.jpg'
        
        if os.path.exists(self.image):
            im = cv2.imread(self.image)
            gt_points = self.data_gt.iloc[:, 2:4].values.tolist()
            gt_future_points = self.data_gt_future.iloc[:, 2:4].values.tolist()
            
            for g in self.data_pred:
                s_past = None
                for j in g:
                    #print(((torch.Tensor(gt_future_points).to(self.device) - j) ** 2).mean())
                    for s in j:
                        if s_past is not None:
                            im = cv2.line(im, (int(s_past[0]), int(s_past[1])), (int(s[0]), int(s[1])), color=(0, 255, 0), thickness=2)
                        s_past = s
            
            g_past = None
            for g in gt_points:
                if g_past is not None:
                    im = cv2.line(im, (int(g_past[0]), int(g_past[1])), (int(g[0]), int(g[1])), color=(255, 255, 0), thickness=2)
                g_past = g
            
            g_past = None
            for g in gt_future_points:
                if g_past is not None:
                    im = cv2.line(im, (int(g_past[0]), int(g_past[1])), (int(g[0]), int(g[1])), color=(0, 0, 255), thickness=2)
                g_past = g
            
            cv2.imwrite(im_path, im)
    
class painter2():
    def __init__(self, image, data, bbox):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image = image
        self.data = data
        self.data_gt_past = bbox[0]
        self.data_gt_future = bbox[1]
        self.data_pred = bbox[2]
        
        self.viz_bbox()
    
    def viz_bbox(self):
        im_path = 'C:\\Users\\user\\Desktop\\' + self.data.sceneId.unique()[0] + '_bbox.jpg'
        
        if os.path.exists(self.image):
            im = cv2.imread(self.image)
            
            for g in self.data_pred:
                for s in g:
                    im = cv2.rectangle(im, (int(s[0]), int(s[1])), (int(s[2]), int(s[3])), color=(0, 255, 0), thickness=1)
            
            for g in self.data_gt_past:
                for s in g:
                    im = cv2.rectangle(im, (int(s[0]), int(s[1])), (int(s[2]), int(s[3])), color=(255, 255, 0), thickness=1)
            
            for g in self.data_gt_future:
                for s in g:
                    im = cv2.rectangle(im, (int(s[0]), int(s[1])), (int(s[2]), int(s[3])), color=(0, 0, 255), thickness=1)
            
            cv2.imwrite(im_path, im)
    