import tensorflow as tf
import numpy as np 
import configs.configs_ML100K as configs
from model import NMC
from data_loader import DataLoader
import time 
import pandas as pd
import os


#command line arguments
FLAGS = tf.app.flags.FLAGS
tf.flags.DEFINE_string("data_dir", "./Bio_Dataset/", "Data directory.")
tf.flags.DEFINE_string("output_basedir", "./outputs/", "Directory for saving and loading model checkpoints.")
tf.flags.DEFINE_string("pretrained_fname", "", "Name of the pretrained model checkpoints (to resume from)")
tf.flags.DEFINE_string("output_dir", "saved_model", "Model output directory.")
tf.flags.DEFINE_string("output_matrix_dir", "./Experiment1/", "Output Matrix Directory")
FLAGS.output_dir = FLAGS.output_basedir + 'snapshots/snapshot'

cfgs = configs.CONFIGS

#my code
#reconstruct the complete matrix from all the batches
def add_batch(complete_matrix, batch, x_indices, y_indices) :

    #print("complete_matrix: ",complete_matrix.shape)
    #print("batch: ",batch.shape)
    #print("x_indices: ", x_indices.shape)
    #print("y_indices: ", y_indices.shape)

    for i in range(len(x_indices)) :

        for j in range(len(y_indices)) :

            #print("xi: ", x_indices[i])
            #print("yi", y_indices[j])

            complete_matrix[x_indices[i], y_indices[j]] = batch[i,j]
            #print("complete_matrix: ", complete_matrix[x_indices[i], y_indices[j]])
            #print("batch: ", batch[i,j])
    

def main(unused_argv):
    assert FLAGS.output_dir, "--output_dir is required"
    # Create training directory.
    output_dir = FLAGS.output_dir
    if not tf.gfile.IsDirectory(output_dir):
        tf.logging.info("Creating training directory: %s", output_dir)
        tf.gfile.MakeDirs(output_dir)

    dl = DataLoader(FLAGS.data_dir)
    dl.load_data()
    dl.split()
    #print(dl.R_tr_unnormalized.shape,"\n\n")
    #print(dl.R_val_unnormalized.shape,"\n\n")
    #print(dl.val_mask.shape,"\n\n")


    x_dim = dl.get_X_dim()
    y_dim = dl.get_Y_dim()

    #print(dl.max_val)
    #print(dl.min_val)

    recons_complete_matrix = np.ones(shape = dl.R.shape, dtype = 'float')
    recons_complete_matrix = np.multiply(recons_complete_matrix, -1)
    R_norm_complete_matrix = np.ones(shape = dl.R.shape, dtype = 'float')
    R_norm_complete_matrix = np.multiply(R_norm_complete_matrix, -1)
    mask_complete_matrix = np.ones(shape = dl.R.shape, dtype = 'float')
    mask_complete_matrix = np.multiply(R_norm_complete_matrix, -1)
    #print(recons_complete_matrix)

    #print("Data",dl)
    #print(x_dim,y_dim)

    #Creating directory for storing R batches and recon bathces in it
    '''if os.path.exists("./BioData_recons") == False :
        os.mkdir("./BioData_recons")'''

    #np.set_printoptions(threshold = np.inf)
    #print(dl.R_tr_unnormalized)


    # Build the model.
    tf.reset_default_graph()
    model = NMC(x_dim, y_dim, cfgs)
    

    if FLAGS.pretrained_fname:
        try:
            print('Resume from %s' %(FLAGS.pretrained_fname))
            model.restore(FLAGS.pretrained_fname)
        except:
            print('Cannot resume model... Training from scratch')
    
    lr = cfgs.initial_lr #learning rate
    epoch_counter = 0
    ite = 0
    while True:
        start = time.time()
        #x-row vector
        #y-column vector

        x, y, R, mask, flag, x_indices, y_indices= dl.next_batch(cfgs.batch_size_x, cfgs.batch_size_y, 'train')
        
        #x, y, R, mask, flag = dl.next_batch(cfgs.batch_size_x, cfgs.batch_size_y, 'val')
        #print(R.shape)
        #print("x.shape ",x.shape,"\n","y.shape ",y.shape,"\n\n")
        
        if np.sum(mask) == 0:
            continue

        load_data_time = time.time() - start
        if flag: 
            epoch_counter += 1

        
        # some boolean variables    
        do_log = (ite % cfgs.log_every_n_steps == 0) or flag
        do_snapshot = flag and epoch_counter > 0 and epoch_counter % cfgs.save_every_n_epochs == 0

        # train one step
        start = time.time()
        loss, recons, ite = model.partial_fit(x, y, R, mask, lr)


        one_iter_time = time.time() - start

        """        np.set_printoptions(threshold = np.inf)
        print("------------------------------R--------------------------")
        print(R,"\n\n")
        print("------------------------------Recon-----------------------")
        print(recons,"\n\n\n\n")"""

        #print("recons.shape ",recons.shape)
        #print("loss.shape",loss.shape)
        #print("ite",ite)

        #Saving R and recon 100*170 batches as npz for visualization
        #file_R = str("./BioData_recons/") + str("R")+str(ite)
        #np.save(file_R,R)

        #file_recons = str("./BioData_recons/") + str("recons") + str(ite)
        #np.save(file_recons,recons)

        #Compiling all the recons  batches in one matrix
        add_batch(recons_complete_matrix, recons, x_indices, y_indices)
        add_batch(R_norm_complete_matrix, R, x_indices, y_indices)
        add_batch(mask_complete_matrix, mask, x_indices, y_indices)



        
        # writing outs

        if do_log:
            print('Iteration %d, (lr=%f) training loss  : %f' %(ite, lr, loss))

        if do_snapshot:
            print('Snapshotting')
            model.save(FLAGS.output_dir)
        
        if flag: 
            # decay learning rate during training
            if epoch_counter % cfgs.num_epochs_per_decay == 0:
                lr = lr * cfgs.lr_decay_factor
                print('Decay learning rate to %f' %lr)
            if epoch_counter == FLAGS.n_epochs:
                if not do_snapshot:
                    #pd.DataFrame(R).to_csv("R_batch.csv")
                    #pd.DataFrame(recons).to_csv("recons_batch.csv")

                    print('Final snapshotting')
                    model.save(FLAGS.output_dir)
                    #np.save("/home/safeer/Documents/QCRI/QCRI-Deep-Learning/Codes-myVersion/neural-matrix-completion-master/BioData_recons_complete_matrix", recons_complete_matrix)
                    #np.save("/home/safeer/Documents/QCRI/QCRI-Deep-Learning/Codes-myVersion/neural-matrix-completion-master/BioData_R_norm_complete_matrix", R_norm_complete_matrix)

                    if not os.path.exists("./" + FLAGS.output_matrix_dir) :
                        os.mkdir("./" + FLAGS.output_matrix_dir)

                    #np.save("./" + FLAGS.output_matrix_dir + "/R_norm_complete_matrix" , R_norm_complete_matrix)
                    #np.save("./" + FLAGS.output_matrix_dir + "/recons_complete_matrix" , recons_complete_matrix)
                    #np.save("./" + FLAGS.output_matrix_dir + "/mask_complete_matrix" , mask_complete_matrix)
                break


if __name__ == '__main__':
    tf.app.run()