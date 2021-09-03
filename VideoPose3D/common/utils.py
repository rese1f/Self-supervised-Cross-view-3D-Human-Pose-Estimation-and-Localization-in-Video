import torch


def sk_len(pose):
    """computing the total length

    Args:
        pose ([tensor]): [v,n,f,17,3]
        
    return:
        length ([tensor]): [v,n,f,1]
    """
    JC = {
        1:0,
        2:1,
        3:2,
        4:0,
        5:4,
        6:5,
        7:0,
        8:7,
        9:8,
        10:9,
        11:8,
        12:11,
        13:12,
        14:8,
        15:14,
        16:15
    }
    length = torch.mean(torch.stack([torch.norm(pose[:,:,:,i]-pose[:,:,:,JC[i]],dim=-1) for i in range(1,17)],dim=-1),dim=-1)
    return length