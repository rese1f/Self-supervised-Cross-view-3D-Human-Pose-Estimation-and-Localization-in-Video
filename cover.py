import torch
#import threading

class cover():
    def __init__(self, data_2d_std, head_index, body_index, leg_index, arm_index):
        # data_2d_std -> [n,x,32,3]

        self.n = data_2d_std.shape[0]
        self.x = data_2d_std.shape[1]
        self.m = data_2d_std.shape[2]

        # self.data -> [x,n,32,3]
        self.data = data_2d_std.transpose(0,1)

        # implement code to change (x,y,z) to (x,z,y) for self.data

        # self.head -> [x,n,3]
        head = self.data[:,:,head_index,:]
        self.head_endpoint = torch.stack((head[:,:,0],head[:,:,2]),2)
        self.head_depth = torch.unsqueeze(head[:,:,1],1)

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
        
        # [x,1,m]
        self.body_depth = torch.unsqueeze(self.get_depth(body),1)
        self.leg_depth = torch.unsqueeze(self.get_depth(leg),1)
        self.arm_depth = torch.unsqueeze(self.get_depth(arm),1)

        # self.data -> [x,32*n,3]
        self.data = self.data.reshape(self.x, self.n*self.m,3)

        # self.data_pos -> [x,32*n,2]
        self.data_pos = torch.stack((self.data[:,:,0],self.data[:,:,2]),2)

        # self.data_depth -> [x,32*n,1]
        self.data_depth =  torch.unsqueeze(self.data[:,:,1],2)

        # record cover at x frame and m vertex
        self.record = torch.zeros((self.x,self.m*self.n))


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

    
    def get_head_cases(self):
        cases = (self.head_depth.expand(self.x,3*self.m,self.head_depth.shape[2])-self.data_depth < 0).nonzero()
        
        for i in range(cases.shape[0]):

            x = cases[i][0]
            v = cases[i][1]

            if self.record[x][v] == 1:                    
                continue
                
            c = cases[i][2]

            point = self.data_pos[x][v]
            head = self.head_endpoint[x][c]

            if torch.sum((point-head)*(point-head)) < 10000:
                self.record[x][v] = 1
        

    def get_body_cases(self):
        cases = (self.body_depth.expand(self.x,3*self.m,self.body_depth.shape[2])-self.data_depth < 0).nonzero()
        
        
        for i in range(cases.shape[0]):

            x = cases[i][0]
            v = cases[i][1]

            if self.record[x][v] == 1:                    
                continue
                
            c = cases[i][2]

            AB = self.body_vector[x][c]
            AP = self.data_pos[x][v] - self.body_endpoint[x][c]
            cross = torch.abs(AB[0]*AP[1]-AB[1]*AP[0])
            LAB = self.body_norm[x][c]
            LAP = torch.norm(AP,2)
            dis = cross / LAB
            k = torch.sum(AP*AB)/LAB/LAB
            if dis < 200 and 0<k<1:
                self.record[x][v] = 1
            

    def get_leg_cases(self):
        cases = (self.leg_depth.expand(self.x,3*self.m,self.leg_depth.shape[2])-self.data_depth < 0).nonzero()
        
        
        for i in range(cases.shape[0]):

            x = cases[i][0]
            v = cases[i][1]

            if self.record[x][v] == 1:                    
                continue
                
            c = cases[i][2]

            AB = self.leg_vector[x][c]
            AP = self.data_pos[x][v] - self.leg_endpoint[x][c]
            cross = torch.abs(AB[0]*AP[1]-AB[1]*AP[0])
            LAB = self.leg_norm[x][c]
            LAP = torch.norm(AP,2)
            dis = cross / LAB
            k = torch.sum(AP*AB)/LAB/LAB
            if dis < 80 and 0<k<1:
                self.record[x][v] = 1
    
    def get_arm_cases(self):
        cases = (self.arm_depth.expand(self.x,3*self.m,self.arm_depth.shape[2])-self.data_depth < 0).nonzero()
        
        
        for i in range(cases.shape[0]):

            x = cases[i][0]
            v = cases[i][1]

            if self.record[x][v] == 1:                    
                continue
                
            c = cases[i][2]

            AB = self.arm_vector[x][c]
            AP = self.data_pos[x][v] - self.arm_endpoint[x][c]
            cross = torch.abs(AB[0]*AP[1]-AB[1]*AP[0])
            LAB = self.arm_norm[x][c]
            LAP = torch.norm(AP,2)
            dis = cross / LAB
            k = torch.sum(AP*AB)/LAB/LAB
            if dis < 50 and 0<k<1:
                self.record[x][v] = 1

    def run(self):
        '''
        t1 = threading.Thread(target=self.get_head_cases)
        t2 = threading.Thread(target=self.get_body_cases)
        t3 = threading.Thread(target=self.get_leg_cases)
        t4 = threading.Thread(target=self.get_arm_cases)

        t1.start()
        t2.start()
        t3.start()
        t4.start()

        t1.join()
        t2.join()
        t3.join()
        t4.join()
        '''

        self.get_head_cases()
        self.get_body_cases()
        self.get_leg_cases()
        self.get_arm_cases()
        
        # cover_std -> [n,x,32]

        return self.record.reshape(self.x,self.n,self.m).transpose(0,1)
