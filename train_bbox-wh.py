import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy
from tqdm import tqdm

mode = 'jaad' # [jaad, pie]
obs_len = 15
pred_len = 45
total_len = obs_len + pred_len
torch.manual_seed(1) # seed

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

train_data = pd.read_pickle(f'./data/{mode}/train_{mode}.pkl')
val_data = pd.read_pickle(f'./data/{mode}/test_{mode}.pkl')

'''
Process train data
'''
X, Y = [], []
for meta_id, meta_df in tqdm(train_data.groupby("metaId", as_index=False)):
    X.append(pd.concat([meta_df["x"][:total_len],meta_df["y"][:total_len],meta_df["w"][:obs_len],meta_df["h"][:obs_len]], ignore_index=True))
    Y.append(pd.concat([meta_df["w"][obs_len:total_len],meta_df["h"][obs_len:total_len]], ignore_index=True))

X_train = np.array(pd.concat(X, ignore_index=True, axis=1).T)
y_train = np.array(pd.concat(Y, ignore_index=True, axis=1).T)

'''
Process val data
'''
X, Y = [], []
for meta_id, meta_df in tqdm(val_data.groupby("metaId", as_index=False)):
    X.append(pd.concat([meta_df["x"][:total_len],meta_df["y"][:total_len],meta_df["w"][:obs_len],meta_df["h"][:obs_len]], ignore_index=True))
    Y.append(pd.concat([meta_df["w"][obs_len:total_len],meta_df["h"][obs_len:total_len]], ignore_index=True))

X_test = np.array(pd.concat(X, ignore_index=True, axis=1).T)
y_test = np.array(pd.concat(Y, ignore_index=True, axis=1).T)

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
print(X_train.shape, X_test.shape)

'''
Model
'''
class FC_WH(nn.Module):
    def __init__(self, input_size, output_size, drop_rate=0.):
        super().__init__()
        mid_size = int((input_size+output_size)/2)
        
        self.fc1 = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.Linear(input_size, input_size),
            nn.Linear(input_size, input_size)
        )
        self.fc2 = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.GELU(),
            nn.Linear(input_size, mid_size),
            nn.Linear(mid_size, mid_size)
        )
        self.fc3 = nn.Sequential(
            nn.LayerNorm(mid_size),
            nn.GELU(),
            nn.Linear(mid_size, output_size),
            nn.Linear(output_size, output_size)
        )
        self.fc4 = nn.Sequential(
            nn.LayerNorm(output_size),
            nn.GELU(),
            nn.Linear(output_size, output_size),
            nn.Linear(output_size, output_size),
            nn.Dropout(drop_rate),
            nn.Linear(output_size, output_size),
        )
        self.trans1 = nn.Linear(input_size, mid_size)
        self.trans2 = nn.Linear(mid_size, output_size)
    
        # Init weight after defined all layers
        self.apply(self._init_weights)
    
    def update_dropout(self, drop_rate):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.p = drop_rate
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.fc1(x) + x
        x = self.fc2(x) + self.trans1(x)
        x = self.fc3(x) + self.trans2(x)
        x = self.fc4(x) + x
        return x

'''
Strat train
'''
early_drop = True # If true, you need to search appropriate drop parameter
batch_size = 128
eval_batch = 'all' # [type(str): 'all', type(int): batch_size]
input_size = obs_len*4 + pred_len*2 # x,y,w,h * 15 + x,y * 45
output_size = pred_len*2 # w,h * 45

if mode == 'jaad':
    EndDrop_epoch = 500
    drop_rate = 0.05
    n_epochs = 4000
elif mode == 'pie':
    EndDrop_epoch = 900
    drop_rate = 0.08
    n_epochs = 8000

if not early_drop:
    EndDrop_epoch = 0
    drop_rate = 0 # If you want Standard-Dropout, then edit this value
else:
    print(f"Using early-dropout {drop_rate} for {EndDrop_epoch} epochs")

model = FC_WH(input_size, output_size, drop_rate)
model.to(device)

criterion_mse = nn.MSELoss()
criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
early_drop_schedule = np.linspace(drop_rate, 0, EndDrop_epoch)

best_mse = np.inf
best_weights = None
history = []

print('Start training')
for epoch in tqdm(range(n_epochs), desc='Epoch'):
    train_loss = []
    model.train()
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
    
    # Update dropout rate
    if epoch < EndDrop_epoch:
        model.update_dropout(early_drop_schedule[epoch])
    
    #print(f"Train MSE: {torch.tensor(train_loss).mean().item()}")
    
    # val model
    model.eval()
    mse = []
    with torch.no_grad():
        if eval_batch == 'all':
            y_pred = model(X_test)
            mse = criterion_mse(y_pred, y_test).item()
        else:
            for i in range(0, len(X_test), eval_batch):
                X_batch = X_test[i:i+eval_batch]
                y_batch = y_test[i:i+eval_batch]
                y_pred = model(X_batch)
                loss = ((y_pred - y_batch) ** 2).mean(dim=1) # MSE
                mse.append(loss)
            mse = torch.cat(mse).mean().item()
        
        #print(f"Val MSE: {round(mse, 3)}")
        history.append(mse)
        if mse < best_mse:
            best_epoch = epoch
            best_mse = mse
            best_weights = copy.deepcopy(model.state_dict())
        
        '''
        # Early Stop
        if epoch - best_epoch > 2000:
            break
        '''

print("MSE: %.2f" % round(best_mse, 3))
print("Best epoch: ", best_epoch)

model.load_state_dict(best_weights)
jit = torch.jit.trace(model, torch.rand(1, input_size).to(device))
jit.save(f'./pretrained_models/bbox_wh/{mode}_wh-{best_epoch}-{int(best_mse)}.pth')

'''
plt.plot(history)
plt.savefig(f'{mode}_learn_{EndDrop_epoch}.png')
#plt.show()
plt.clf()
'''
