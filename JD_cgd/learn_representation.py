import numpy as np
import scipy.sparse as sparse
import nimfa
import logging

dir = 'data/'
def read_matrix(fname, suffix = ''):
    I = []
    J = []
    V = []
    user_dict = dict( )
    sku_dict = dict( )
    array_dict = set( )
    with open(fname) as fin:
        fin.readline( )
        for line in fin:
            terms = line.split(',')
            user_id = int(float(terms[1]))
            sku_id = int(terms[2])
            if user_id not in user_dict:
                user_dict[user_id] = len(user_dict)
            if sku_id not in sku_dict:
                sku_dict[sku_id] = len(sku_dict)
            i = user_dict[user_id]
            j = sku_dict[sku_id]
            #在准备数据时已经去重
            # if (i,j) not in array_dict:
            I.append( i )
            J.append( j )
            V.append( 1 )
                # array_dict.add( (i,j) )
    array_dict = None
    I = np.array( I )
    J = np.array( J )
    V = np.array( V )
    A = sparse.coo_matrix( (V,(I,J) ) , shape=(len(user_dict),len(sku_dict) ) )
    print('read matrix done. start to fit.')
    #divergence or euclidean; fro div coon
    nmf = nimfa.Nmf(A, max_iter=200, rank=30, update='euclidean', objective='fro')
    nmf_fit = nmf( )
    print('Euclidean distance: %5.3f' % nmf_fit.distance(metric='euclidean'))
    W = nmf_fit.basis( ).todense( )
    H = nmf_fit.coef( ).todense( ).T
    print(W.shape)
    print(H.shape)
    #decode and save
    with open(dir+'user_{}.mat'.format(suffix),'w') as fout:
        head = ['user_id']
        for i in range(W.shape[1]):
            head.append('u_'+str(i))
        fout.write(','.join(head)+'\n')
        dict_res = dict()
        for uid, index in  user_dict.items():
            dict_res[index] = uid
        for index in range(len(W)):
            row = [str( dict_res[index] )]
            for j in range(W.shape[1]):
                row.append( str(W[index,j]) )
            fout.write(','.join(row)+'\n')
    print('save user matrix done.')
    with open(dir+'sku_{}.mat'.format(suffix),'w') as fout:
        head = ['sku_id']
        for i in range(H.shape[1]):
            head.append('sku_'+str(i))
        fout.write(','.join(head)+'\n')
        dict_res = dict()
        for sku_id, index in sku_dict.items():
            dict_res[index] = sku_id
        for index in range(len(H)):
            row = [str(dict_res[index] ) ]
            for j in range(H.shape[1]):
                row.append( str(W[index,j]) )
            fout.write(','.join(row)+'\n')
    print('save item matrix done.')

if __name__ == '__main__':
    logging.basicConfig(format='%(asctimes)s %(name)-1s %(levelname)-1s %(message)s', level=logging.INFO)
    read_matrix(dir+'up_matrix_4.csv','4')
    read_matrix(dir + 'up_matrix_1.csv', '1')
    read_matrix(dir + 'up_matrix_2.csv', '2')