import torch.utils.data
from util import *
from torch.utils.data import DataLoader
from dataset import dataset_loader
from model.model_full import MDEDNet

if __name__ == '__main__':    
    model_path = os.path.join('checkpoints/epoch_131.pth')
    device = torch.device('cuda:0') 
    model = MDEDNet().to(device)
    print('model parameters: [%.2f] M'%(sum(param.numel() for param in model.parameters())/1e6))
    model.load_state_dict(torch.load(model_path))
    model.eval()

    save_path_deblur  = os.path.join('results/deblur') 
    save_path_denoise  = os.path.join('results/denoise') 
    if not os.path.exists(save_path_deblur):
        os.makedirs(save_path_deblur)  
    if not os.path.exists(save_path_denoise):
        os.makedirs(save_path_denoise)  
                           
    test_dataset = dataset_loader()
    test_loader = DataLoader(dataset=test_dataset, num_workers=8, batch_size=1)       

    for iteration, data in enumerate(test_loader):
        with torch.no_grad():
            blur = data['blur'].to(device)
            sharp = data['sharp'].to(device)
            event_noise = data['event_noise'].to(device)
            deblur,event_denoise = model(blur,event_noise)
            name = data['name']
            save_path_denblur_ = os.path.join(save_path_deblur,name[0]+'.png') 
            save_path_denoise_ = os.path.join(save_path_denoise,name[0]+'.png')             
            img_save(deblur,save_path_denblur_) 
            save_tensor_to_npy(event_denoise,save_path_denoise_) 
    