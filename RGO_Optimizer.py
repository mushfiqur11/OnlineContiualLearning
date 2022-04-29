import tensorflow as tf
import numpy as np

class RGO_Optimizer:
    def __init__(self, network,alpha=0.01,gamma=1.0,optimizer=tf.train.AdamOptimizer(0.01)):
        self.network=network
        self.optimizer = optimizer
        # self.ortho_method='cos2'

        #divide epochs
        self.alpha=tf.Variable([alpha],dtype=tf.float32)
        self.alpha0=1
        self.beta0=0
        self.gamma0=1.0

        self.shape_list=[]
        self.size_list=[]
        self.n_vars=0
        self.n_paras=0
        self.grad_list=[]
        self.size_list=[]
        self.dim_list=[]

        self.initialized=False

    def minimize(self, loss,var_list=None):
        grad_vars_loss = self.optimizer.compute_gradients(loss,var_list=var_list)

        if not self.initialized:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                
                for grad, _ in grad_vars_loss:
                    self.shape_list.append(sess.run(tf.shape(grad)))
                    self.size_list.append(sess.run(tf.size(grad)))
                    self.dim_list.append(len(self.shape_list[-1]))
                    self.n_vars+=1

            self.gamma=tf.Variable(1.0,dtype=tf.float32)
            self.beta=tf.Variable(0.0,dtype=tf.float32)
            self.n=tf.Variable(0,dtype=tf.int32)

            self.p_size_list=[]
            for i in range(0,self.n_vars):
                if self.dim_list[i]<=2:
                    self.p_size_list.append(self.shape_list[i][0])
                elif self.dim_list[i]==3:
                    self.p_size_list.append(self.shape_list[i][0]*self.shape_list[i][1])
                else:
                    self.p_size_list.append(self.shape_list[i][0]*self.shape_list[i][1]*self.shape_list[i][2])
            
            self.P_list=[tf.Variable(tf.eye(i)) for i in self.p_size_list]

            self.P_total_trace=tf.reduce_sum([tf.trace(P) for P in self.P_list])

            self.P_total_dim=np.sum(self.p_size_list)

            self.initialized=True
        
        grad_loss_list=[]
        for grad, _ in grad_vars_loss:
            grad_loss_list.append(grad)
            
        grad_loss_list = [self.apply_P(P,g,shape) for (P,g,shape) in zip(self.P_list,grad_loss_list,self.shape_list)]

        self.grad_vars=[]
        for i,(_,v) in enumerate(grad_vars_loss):
            self.grad_vars.append((grad_loss_list[i],v))

        op = [self.optimizer.apply_gradients(self.grad_vars)]

        return op
    def apply_P(self,P,g,shape):
        #TODO 
        #return g
        if len(shape)==3:
            return tf.reshape(tf.matmul(P/tf.trace(P)*shape[0]*shape[1],tf.reshape(g,[shape[0]*shape[1],shape[2]])),shape=shape)

        elif len(shape)>3:
            return tf.reshape(
                tf.matmul(P/tf.trace(P)*shape[0]*shape[1]*shape[2],tf.reshape(g,shape=[shape[0]*shape[1]*shape[2],shape[3]])),shape=shape)

        else:
            if len(shape)==1:
                return tf.reshape(tf.matmul(P/tf.trace(P)*shape[0],tf.reshape(g,[-1,1])),shape=shape)
                #return tf.reshape(tf.matmul(P,tf.reshape(g,[-1,1])),shape=shape)

            LayerNormalize=True
            if LayerNormalize:
                return tf.matmul(P/tf.trace(P)*shape[0],g)
            return tf.matmul(P/self.P_total_trace*self.P_total_dim,g)
            #return tf.matmul(P,g)


    def update(self,network,y,var_list=None):

        y_=tf.where(tf.reduce_sum(y,axis=0,keep_dims=True)>0,x=tf.ones([1,np.shape(y)[1]]),y=tf.zeros([1,np.shape(y)[1]]))
        a=tf.exp(network)*y_/tf.reduce_sum(tf.exp(network)*y_,axis=1,keep_dims=True)
        grad_vars_net=self.optimizer.compute_gradients(network*y* tf.stop_gradient(tf.sqrt(a-a*a)),var_list=var_list)

        grad_net_list=[]
        for i,(grad, _) in enumerate(grad_vars_net):
            grad_net_list.append(tf.reshape(
                tf.reduce_mean(grad,keep_dims=True,
                axis=list(range((lambda x : 1 if x==1 else x-1 if x<3 else 3)(self.dim_list[i]),self.dim_list[i]))),[-1,1]))

        P_list=self.P_list

        k_list=[tf.matmul(P, g) for P,g in zip(P_list,grad_net_list)]

        delta_P_list=[tf.square(tf.cos(self.beta*i)) * tf.divide(tf.matmul(k, tf.transpose(k)), self.alpha *tf.pow(self.gamma,i)+tf.square(tf.cos(self.beta*i))* tf.matmul(tf.transpose(g), k)) for k,g,i in zip(k_list,grad_net_list,list(range(self.n_vars)))]

        P_list = [tf.assign_sub(P, delta_P) for P,delta_P in zip(P_list,delta_P_list)]

        op = [tf.assign(self.P_list[i],P_list[i]) for i in range(len(P_list))]+[tf.assign(self.gamma,self.gamma*self.gamma0),tf.assign(self.beta,self.beta+self.beta0),tf.assign(self.n,self.n+1),]
        return op

    def minimize_and_update(self, loss,y,var_list=None):
        return self.minimize(self,loss,var_list=var_list)+self.update(y,var_list=var_list)