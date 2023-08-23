import torch
import torch.nn.functional as F
def euclidean_dist(x, y):
    # x: N x D
    # y: M x D

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def sisnr(x, s, n_shot,eps=1e-8):
    """
    calculate training loss
    input:
          x: separated signal, N x S tensor
          s: reference signal, N x S tensor
    Return:
          sisnr: N tensor
    """
    n_class = s.size(0)
    d = x.size(1)
    x = x.view(n_class,n_shot,d)
    #s = s.unsqueeze(1).repeat(1,n_shot,1)
    s = s.unsqueeze(1).repeat(1,n_shot,1)
    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                x.shape, s.shape))
    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    #x_zm = x
    #s_zm = s
    t = torch.sum(
        x_zm * s_zm, dim=-1,
        keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
    #intra_class = -torch.mean(20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps)))
    intra_class = torch.mean(2 * torch.log10(eps +  l2norm(x_zm - t)/ (l2norm(t) + eps))) #2
    #intra_class = torch.mean((l2norm(x_zm - t)/(eps + l2norm(t)))**2) #3
    #intra_class = torch.mean(l2norm(x_zm - t)/(eps + l2norm(t))) #4
    #intra_class = torch.mean(2 * torch.log(eps + l2norm(t) / (l2norm(x_zm - t) + eps)))
    inter_class = 0
    for _ in range(n_class-1):
        s_zm = torch.cat([s_zm[1:],s_zm[0].unsqueeze(0)],dim=0)
        t = torch.sum(
            x_zm * s_zm, dim=-1,
            keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
        #inter_class += torch.mean(20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps)))
        inter_class += torch.mean(2 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))) #2
        #inter_class += torch.mean((l2norm(t)/(eps + l2norm(x_zm - t)))**2) #3
        #inter_class += torch.mean(l2norm(t)/(eps + l2norm(x_zm - t))) # 4
        #inter_class += torch.mean(2 * torch.log(eps + l2norm(t) / (l2norm(x_zm - t) + eps)))
    #return torch.mean(20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps)))
    #return intra_class - inter_class/(n_class-1)
    return intra_class + inter_class/(n_class-1), intra_class, inter_class

def sisnrv2(x, s, n_shot,eps=1e-8):
    """
    calculate training loss
    input:
          x: separated signal, N x S tensor
          s: reference signal, N x S tensor
    Return:
          sisnr: N tensor
    """
    n_class = s.size(0)
    d = x.size(1)
    x = x.view(n_class,n_shot,d)
    p = s
    s = s.unsqueeze(1).repeat(1,n_shot,1)


    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                x.shape, s.shape))
    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    
    t = torch.sum(
        x_zm * s_zm, dim=-1,
        keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
    #intra_class = torch.mean(20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps)))
    intra_class = torch.mean(2 * torch.log10(eps +  l2norm(x_zm - t)/ (l2norm(t) + eps))) #2
    #intra_class = torch.mean((l2norm(x_zm - t)/(eps + l2norm(t)))**2) #3
    #intra_class = torch.mean(l2norm(x_zm - t)/(eps + l2norm(t))) #4
    #intra_class = torch.mean(2 * torch.log(eps + l2norm(t) / (l2norm(x_zm - t) + eps)))
    inter_class = 0
    for i in range(n_class-1):
        x_zm = p[i]
        for j in range(i+1,n_class):
            s_zm = p[j]
            t = torch.sum(
                x_zm * s_zm, dim=-1,
                keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
            inter_class += torch.mean(2 * torch.log10( l2norm(t) / (l2norm(x_zm - t) + eps)))

        #inter_class += torch.mean((l2norm(t)/(eps + l2norm(x_zm - t)))**2) #3
        #inter_class += torch.mean(l2norm(t)/(eps + l2norm(x_zm - t))) # 4
        #inter_class += torch.mean(2 * torch.log(eps + l2norm(t) / (l2norm(x_zm - t) + eps)))
    #return torch.mean(20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps)))
    #return intra_class - inter_class/(n_class-1)
    return intra_class + inter_class/(n_class-1),intra_class,0.1*inter_class


def cos_similarity(x, y):
    # x: N x D
    # y: M x D

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    #t = torch.sum(x*y,dim=-1,keepdim=True) / (torch.norm(x,dim=-1,keepdim=True)*torch.norm(y,dim=-1,keepdim=True))

    return F.cosine_similarity(x,y,dim=-1)