#CODE BY M M AKHTAR
from src.model_hat_csi import HAT_CNN
from src.nt_xent_loss import nt_xent_loss
from src.utils import get_data_loaders, get_contrastive_views
import torch
import torch.optim as optim
import torch.nn.functional as F

model = HAT_CNN(num_tasks=2) 
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_loader, test_loader = get_data_loaders()

for epoch in range(10):  
    running_loss = 0.0
    task_id = 0  
    
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        view1, view2 = get_contrastive_views(inputs)

        z1 = model(view1, task_id)
        z2 = model(view2, task_id)

        outputs = model(inputs, task_id)

        ce_loss = F.cross_entropy(outputs, labels)

        contrastive_loss = nt_xent_loss(z1, z2)

        total_loss = ce_loss + contrastive_loss

        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()
        if i % 100 == 99: 
            print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100}')
            running_loss = 0.0

print('Finished Training HAT+CSI')
