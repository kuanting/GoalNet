# GoalNet

## model
GoalNet as shown in paper.
https://arxiv.org/abs/2402.19002

## model_seg
1. The prediction results are more accurate if segmentation is used, but the corresponding segmentation model needs to be trained first, and specify the model_seg.py as GoalNet training structure.
2. <Warning> Note that if the dataset does not provide segmentation data, and you label it yourself, it means that additional training data beyond the dataset is used, and it may not be appropriate to compare this result.

## model_det
GoalNet for deterministic trajectory prediction.
