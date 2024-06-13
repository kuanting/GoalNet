# GoalNet
GoalNet full source code.

GoalNet, a new trajectory prediction neural network based on the goal areas of a pedestrian. GoalNet multi-modal trajectory results significantly improves the previous state-of-the-art performance by 48.7% on the JAAD and 40.8% on the PIE dataset.

GoalNet_Traj.
![GoalNet_Backbone](https://github.com/28598519a/GoalNet/assets/33422418/60dc91ae-e2a2-430c-9ab0-14115f7003a3)

GoalNet_bbox.
![GoalNet_bbox](https://github.com/28598519a/GoalNet/assets/33422418/4a6d1412-a305-4d1a-80b3-6b4324c71868)

## model
GoalNet as shown in paper.
https://arxiv.org/abs/2402.19002

## model_seg
1. The prediction results are more accurate if segmentation is used, but the corresponding segmentation model needs to be trained first, and specify the model_seg.py as GoalNet training structure.
2. <Warning> Note that if the dataset does not provide segmentation data, and you label it yourself, it means that additional training data beyond the dataset is used, and it may not be appropriate to compare this result.

## model_det
GoalNet for deterministic trajectory prediction.
