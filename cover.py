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
        self.body = self.generate_cover(body_index)
        self.leg = self.generate_cover(leg_index)
        self.arm = self.generate_cover(arm_index)

        # self.data -> [x,32*n,3]
        self.data = self.data.reshape(self.x, self.n*self.m,3)

        # self.cover -> [x,32*n,1] 
        # record the cover, 1 represent cover 
        self.cover = torch.zeros(self.x,self.m*self.n,1)


    def generate_cover(self, index):
        """
        generate the covering tensor
        index -> list<int>
        self.data -> [x,n,32,3]
        return -> [x,m,2,3]
        """
        return torch.stack([self.data[:,:,i,:] for i in index],2).reshape(self.x,self.n,int(len(index)/2),2,3).reshape(self.x,self.n*int(len(index)/2),2,3)
        

    def judge_head(self):
        """
        for single data, judge for every frame
        """
        # implement code


    def judge_body(self):
        """
        for single data, judge for every frame
        """
        # implement code


    def judge_leg(self):
        """
        for single data, judge for every frame
        """
        # implement code


    def judge_arm(self):
        """
        for single data, judge for every frame
        """
        # implement code
    
    
    def main(self):
        self.judge_head()
        self.judge_body()
        self.judge_leg()
        self.judge_arm()
        self.data[:,:,:,-1] = self.cover
        return self.data