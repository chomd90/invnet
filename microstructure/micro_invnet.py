from invnet import BaseInvNet


#proj ablation: /home/km3888/invnet/runs/Feb05_15-49-30_hegde-lambda-1 and the next four after
#running w/ shortest paths instead of longest ; Feb06_20-55-29_hegde-lambda-1 - /Feb06_20-55-57_hegde-lambda-1
#Fixed shortest paths: Feb09_11-53-16 -
#shortest paths with correct proj loss reporting: /Feb10_12-17-04-Feb10_12-18-20
class InvNet(BaseInvNet):

    def __init__(self,batch_size,output_path,data_dir,lr,critic_iters,\
                 proj_iters,hidden_size,device,lambda_gp,edge_fn,max_op,restore_mode=False):

        self.max_i,self.max_j=64,64
        super().__init__(batch_size,output_path,data_dir,lr,critic_iters,proj_iters,64,64,hidden_size,device,lambda_gp,1,edge_fn,max_op,False,restore_mode)

        print(self.proj_iters)
        self.edge_fn=edge_fn
        self.max_op=max_op


    def real_p1(self,images):
        '''
        Images: [b,64,64,6]
        p1: [b,6]
        '''
        lengths=self.dp_layer(images)
        if self.p1_mean is not None:
            lengths=self.normalize_p1(lengths)
        return lengths.view((-1,1))


    def norm_data(self,data):
        #only get used for saving images
        data=data.view(-1,self.max_i,self.max_j)
        mean=data.mean(dim=0)
        deviation=data.std(dim=0)
        return (data-mean)/deviation

    def format_data(self,data):
        return data.view((self.batch_size,64,64))


