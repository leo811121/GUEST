from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data import WeightedRandomSampler
import torch
import numpy as np

# customed dataset
class bertDataset(Dataset):
    def __init__(self, claim_comms, labels, claims, claim_features, comm_features, comm_masks, tweet_struct_temp):
        self.claim_comms = claim_comms
        self.claims = claims
        self.labels = labels
        self.claim_features = claim_features
        self.comm_features = comm_features
        self.comm_masks = comm_masks
        self.tweet_struct_temp = tweet_struct_temp
    def __getitem__(self, index):
        return(self.claim_comms[index],torch.tensor(np.argmax(self.labels[index])),self.claims[index],
               torch.tensor(self.claim_features[index]), torch.tensor(self.comm_features[index]).long(),
               torch.tensor(self.comm_masks[index]).long(), torch.tensor(self.tweet_struct_temp[index]).float(),
               )
        
    def __len__(self):
        return len(self.claims)
    
#Random Sampling
def get_randsampler(vLabels):
        trueCount = 0
        falseCount = 0
        for i in vLabels: 
            if i == [0.0, 1.0]:
                trueCount += 1
            else:
                falseCount += 1
        print(trueCount, falseCount)

        samples_weight = []
        for i in vLabels:
            if i == [0.0, 1.0]:
                samples_weight.append(1./trueCount)
            else:
                samples_weight.append(1./falseCount*1.6)

        RandomSampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
        return RandomSampler