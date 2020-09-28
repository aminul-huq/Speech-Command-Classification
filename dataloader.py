from torch.utils.data import DataLoader,random_split,Dataset

class SpeechDataLoader(Dataset):
    
    def __init__(self,data,labels,list_dir,transform=None):
        self.data = data
        self.labels = labels
        self.label_dict = list_dir
        self.transform = transform
            
    def __len__(self):
        return len(self.data)    
    
    def __getitem__(self,idx):
        waveform = self.data[idx]
        
        if self.transform != None:
            waveform = self.transform(waveform)

        if self.labels[idx] in self.label_dict:
            out_labels = self.label_dict.index(self.labels[idx])
            

        return waveform, out_labels
    
    
    
    
    
    