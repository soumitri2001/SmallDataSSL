import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimCLRLoss(nn.Module):
    def __init__(self, tau=0.5):
        super(SimCLRLoss, self).__init__()
        self.tau = tau
    
    def forward(self, z1, z2):
        B = z1.shape[0]
        out = torch.cat([z1, z2], dim=0) # [2*B, D]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.tau) # [2*B, 2*B]
        
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * B, device=sim_matrix.device)).bool() # [2*B, 2*B] binary mask
        sim_matrix = sim_matrix.masked_select(mask).view(2 * B, -1) # [2*B, 2*B-1] 
        
        pos_sim = torch.exp(torch.sum(z1 * z2, dim=-1) / self.tau) # [B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0) # [2*B]
        
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        return loss


class DCLWLoss(nn.Module):
    def __init__(self, tau=0.1, sigma=0.5):
        super(DCLWLoss, self).__init__()
        self.tau = tau
        self.sigma = sigma
        self.SMALL_NUM = np.log(1e-45)
        self.weight_fn = lambda z1, z2: 2 - z1.size(0) * F.softmax((z1 * z2).sum(dim=1) / self.sigma, dim=0).squeeze()
        
    def forward(self, z1, z2):
        cross_view_distance = torch.mm(z1, z2.t())
        positive_loss = -torch.diag(cross_view_distance) / self.tau
        positive_loss = positive_loss * self.weight_fn(z1, z2)
        neg_similarity = torch.cat((torch.mm(z1, z1.t()), cross_view_distance), dim=1) / self.tau
        neg_mask = torch.eye(z1.size(0), device=z1.device).repeat(1, 2)
        negative_loss = torch.logsumexp(neg_similarity + neg_mask * self.SMALL_NUM, dim=1, keepdim=False)
        return (positive_loss + negative_loss).mean()



class VICRegLoss(nn.Module):
    def __init__(self, l=25, mu=25, nu=1):
        super(VICRegLoss, self).__init__()
        self.l = l
        self.mu = mu
        self.nu = nu
        self.sim_loss = nn.MSELoss()

    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    # variance loss
    def std_loss(self, z_a, z_b):
        std_z_a = torch.sqrt(z_a.var(dim=0) + 1e-04)
        std_z_b = torch.sqrt(z_b.var(dim=0) + 1e-04)
        std_loss = torch.mean(F.relu(1 - std_z_a)) + torch.mean(F.relu(1 - std_z_b))
        return std_loss
    
    # covariance loss
    def cov_loss(self, z_a, z_b):
        N = z_a.shape[0]
        D = z_a.shape[1]
        z_a = z_a - z_a.mean(dim=0)
        z_b = z_b - z_b.mean(dim=0)
        cov_z_a = (z_a.T @ z_a) / (N - 1)
        cov_z_b = (z_b.T @ z_b) / (N - 1)
        cov_loss =self.off_diagonal(cov_z_a).pow_(2).sum() / D + self.off_diagonal(cov_z_b).pow_(2).sum() / D
        return cov_loss

    def forward(self, z1, z2):
        _sim_loss = self.sim_loss(z1, z2)
        _std_loss = self.std_loss(z1, z2)
        _cov_loss = self.cov_loss(z1, z2)
        loss = self.l * _sim_loss + self.mu * _std_loss + self.nu * _cov_loss
        return loss


class BarlowLoss(nn.Module):
    def __init__(self, lambd=0.0051):
        super(BarlowLoss, self).__init__()
        self.lambd = lambd

    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1, z2):
        # empirical cross-correlation matrix
        B = z1.shape[0]
        c = z1.T @ z2
        # sum the cross-correlation matrix between all gpus
        c.div_(B)
        # compute barlow loss
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss


class SimSiamLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(SimSiamLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, z1, z2):
        sim_1 = -(F.normalize(z1, dim=-1) * F.normalize(z2.detach(), dim=-1)).sum(dim=-1).mean()
        sim_2 = -(F.normalize(z2, dim=-1) * F.normalize(z1.detach(), dim=-1)).sum(dim=-1).mean()
        loss = self.alpha * sim_1 + self.beta * sim_2
        return loss


class TiCoLoss(nn.Module):
    def __init__(self, beta=0.9, rho=8):
        super(TiCoLoss, self).__init__()
        self.beta = beta
        self.rho = rho

    def forward(self, C, z1, z2):
        z_1 = F.normalize(z1, dim = -1)
        z_2 = F.normalize(z2, dim = -1)
        B = torch.mm(z_1.T, z_1)/z_1.shape[0]
        C = self.beta * C + (1 - self.beta) * B
        loss = - (z_1 * z_2).sum(dim=1).mean() + self.rho * (torch.mm(z_1, C) * z_1).sum(dim=1).mean()
        return loss, C
