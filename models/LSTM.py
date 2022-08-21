import torch
import torch.nn as nn
import torch.nn.functional as F

def window(x,win_size,stride):
    """
    x: time seq needs to be unfolded. 1d np.array
    win_size: window size.
    stride: stride to take window.
    Return:
        data [m,win_size] np.array
    """
    return torch.tensor(x,dtype=float).unfold(0,win_size,stride).numpy()

class pred_LSTM(nn.Module):
    def __init__(self,in_channel,hidden_channel,num_layers,window,pred_length):
        super().__init__()
        self.pred_len = pred_length
        self.in_channel = in_channel
        self.rnn = nn.LSTM(in_channel,hidden_channel,num_layers,batch_first=True)
        self.cnn1 = nn.Conv1d(in_channels=hidden_channel,out_channels=hidden_channel*2,kernel_size=window // 10,stride=window // 10,padding=in_channel//2)
        self.cnn2 = nn.Conv1d(in_channels=hidden_channel*2,out_channels=hidden_channel*2,kernel_size=10)
        # self.bn1 = nn.BatchNorm1d(hidden_channel*2)
        self.fc = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hidden_channel,hidden_channel),
            nn.Tanh(),
            nn.Linear(hidden_channel,pred_length),
            nn.Dropout(p=0.3)
        )
        self.fc2 = nn.Linear(hidden_channel*2,pred_length)
        self.fc3 = nn.Linear(pred_length,pred_length)

    def forward(self,x):
        # x: [B,win_size]
        batch_size = x.shape[0]
        base_line = x[:,-1].view(batch_size,1).repeat(1,self.pred_len) # [B,P]
        Feat = self.encoder(x)
        # x: [B,win_size-C+1,C]
        Feat,_ = self.rnn(Feat)
        x = Feat[:,-1].reshape(batch_size,-1) # [B,H]
        x = self.fc(x)
        # [B,P]
        Feat = torch.tanh(self.cnn1(torch.tanh(Feat.transpose(2,1))))
        Feat = self.fc2(torch.tanh(self.cnn2(Feat)).view(batch_size,-1))
        x = x + Feat
        # [B,P]
        x = torch.tanh(self.fc3(x)) * base_line + base_line
        return x

    def encoder(self,x):
        # x: [B,win_size]
        win_size = x.shape[1]
        Feat = torch.zeros(x.shape[0],win_size-self.in_channel+1,self.in_channel,device='cuda').float()
        x = x.unfold(dimension=1,size=(win_size-self.in_channel+1),step=1).flip(dims=[1]).transpose(2,1) # [B,win_size-C+1,C]
        Feat[:,:,0] = x[:,:,0]
        Feat[:,:,1] = x[:,:,0] - x[:,:,1]
        Feat[:,:,2] = x[:,:,0] - 2 * x[:,:,1] + x[:,:,2]
        if self.in_channel > 3:
            Feat[:,:,3:] = x[:,:,3:]
        return Feat
        # Feat: [B,win_size-C+1,C]