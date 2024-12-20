import torch 
import numpy as np

x = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)
lr=0.1
epoch=15
class LineerAltanRegressor():
    def __init__(self,x,y):
        self.weight=torch.tensor(0.0,dtype=torch.float32,requires_grad=True)
        self.x=x
        self.y=y
    def forwardpass(self):
        return self.weight*self.x
    #loss strategy:mse
    def lossfunct(self,pred_y):
        loss=((pred_y-y)**2).mean()
        
        return loss
    
    
    def __call__(self, *args, **kwds):
        for i in range(epoch):
            pred_y=self.forwardpass()
            loss=self.lossfunct(pred_y)
            
            loss.backward()
            with torch.no_grad():
                self.weight -= lr * self.weight.grad
            
            # Gradyanı sıfırla
            self.weight.grad.zero_()

            # Her epoch sonunda ağırlık ve kayıp değerini yazdır
            if i % 1 == 0:
                print(f"Epoch {i+1}: Weight = {self.weight.item()}, Loss = {loss.item()}")
            

# Modeli çalıştır
model = LineerAltanRegressor(x, y)
model()