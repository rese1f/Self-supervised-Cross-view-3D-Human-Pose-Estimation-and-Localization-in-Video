import torch

class cover:
    def __init__(self, data_2d_std, head_index, body_index, leg_index, arm_index):
        # data_2d_std -> [n,x,32,3]
        self.n = data_2d_std.shape[0]
        self.x = data_2d_std.shape[1]
        self.m = data_2d_std.shape[2]

        # self.data -> [x,n,32,3]
        self.data = data_2d_std.transpose(0,1)

        # implement code to change (x,y,z) to (x,z,y) for self.data

        # self.head -> [x,n,3]
        self.head = self.data[:,:,head_index,:]

        # self.cover -> [x,m,2,3]
        body = self.get_cover(body_index)
        leg = self.get_cover(leg_index)
        arm = self.get_cover(arm_index)

        # [x,m,2]
        self.body_endpoint = self.get_endpoint(body)
        self.leg_endpoint = self.get_endpoint(leg)
        self.arm_endpoint = self.get_endpoint(arm)

        # [x,m,2]
        self.body_vector = self.get_vector(body)
        self.leg_vector = self.get_vector(leg)
        self.arm_vector = self.get_vector(arm)
        
        # [x,m]
        self.body_norm = self.get_norm(self.body_vector)
        self.leg_norm = self.get_norm(self.leg_vector)
        self.arm_norm = self.get_norm(self.arm_vector)
        
        # [x,m]
        self.body_depth = self.get_depth(body)
        self.leg_depth = self.get_depth(leg)
        self.arm_depth = self.get_depth(arm)

        # self.data -> [x,32*n,3]
        self.data = self.data.reshape(self.x, self.n*self.m,3)

        # self.cover -> [x,32*n,1] 
        # record the cover, 1 represent cover 
        self.cover = torch.zeros(self.x,self.m*self.n,1)


    def get_cover(self, index):
        """
        generate the covering tensor
        index -> list<int>
        self.data -> [x,n,32,3]
        return -> [x,m,2,3]
        """
        return torch.stack([self.data[:,:,i,:] for i in index],2).reshape(self.x,self.n,int(len(index)/2),2,3).reshape(self.x,self.n*int(len(index)/2),2,3)
        

    def get_endpoint(self,data):
        return torch.stack((data[:,:,0,0],data[:,:,0,2]),2)
    

    def get_vector(self,data):
        return torch.stack((data[:,:,1,0],data[:,:,1,2]),2)-torch.stack((data[:,:,0,0],data[:,:,0,2]),2)

    
    def get_norm(self,data):
        return data.norm(3,2)

    
    def get_depth(self,data):
        return torch.max(data[:,:,:,1],2).values