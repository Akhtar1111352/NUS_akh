#CODE BY M M AKHTAR
import torch
import torch.nn.functional as F


def nt_xent_loss(z_i, z_j, temperature=0.5):
    batch_size = z_i.shape[0]
    z = torch.cat([z_i, z_j], dim=0)  
    z = F.normalize(z, dim=1)  

    
    similarity_matrix = torch.matmul(z, z.T)
    labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(z.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    positives = similarity_matrix[labels.bool()].view(batch_size, -1)
    negatives = similarity_matrix[~labels.bool()].view(batch_size, -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(z.device)
    logits = logits / temperature

    return F.cross_entropy(logits, labels)
