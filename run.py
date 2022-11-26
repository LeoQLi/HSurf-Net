import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--mode', type=str, default='', choices=['train', 'test'])
parser.add_argument('--data_set', type=str, default='PCPNet', choices=['PCPNet', 'SceneNN', 'Semantic3D'])
parser.add_argument('--ckpt_dirs', type=str, default='001')
parser.add_argument('--epoch', type=int, default=900)
FLAGS = parser.parse_args()

dataset_root = '/data1/lq/Dataset/'
gpu = FLAGS.gpu
lr = 0.0005
num_knn = 16
train_patch_size = 700
train_batch_size = 100

if FLAGS.mode == 'train':
    trainset_list = 'trainingset_whitenoise.txt'
    decode_knn = num_knn
    resume = ''

    os.system('CUDA_VISIBLE_DEVICES={} python train.py --dataset_root={} --trainset_list={} --patch_size={} --batch_size={} \
                                                    --num_knn={} --decode_knn={} --lr={} --resume={}'.format(
                gpu, dataset_root, trainset_list, train_patch_size, train_batch_size, num_knn, decode_knn, lr, resume))

elif FLAGS.mode == 'test':
    data_set = FLAGS.data_set
    log_root = './log/'
    test_patch_size = train_patch_size
    test_batch_size = 700
    decode_knn = num_knn * 1
    ckpt_dirs = FLAGS.ckpt_dirs
    ckpt_iters = '100-1500-200'  # epoch: start-stop-step
    if FLAGS.epoch > 0:
        ckpt_iters = '%d-1000-200' % FLAGS.epoch

    save_pn = False        # to save the point normals as '.normals' file
    sparse_patches = True  # to output sparse point normals or not, but the evaluation is alway conduct on the sparse point clouds.

    testset_list = None
    eval_list = None
    # testset_list = 'testset_high_noise.txt'
    # eval_list = testset_list

    if data_set == 'SceneNN':
        testset_list = 'testset_SceneNN.txt'
        eval_list = 'testset_SceneNN_clean.txt testset_SceneNN_noise.txt'
    elif data_set == 'Semantic3D':
        testset_list = 'testset_Semantic3D.txt'
        eval_list = testset_list

    command = 'python test.py --gpu={} --dataset_root={} --data_set={} --log_root={} --ckpt_dirs={} --ckpt_iters={} \
                            --patch_size={} --batch_size={} --num_knn={} --decode_knn={} --save_pn={} --sparse_patches={}'.format(
            gpu, dataset_root, data_set, log_root, ckpt_dirs, ckpt_iters, test_patch_size, test_batch_size, num_knn, decode_knn, save_pn, sparse_patches)

    if testset_list is None and eval_list is None:
        os.system('{}'.format(command))
    else:
        os.system('{} --testset_list={} --eval_list {}'.format(command, testset_list, eval_list))

else:
    print('The mode is unsupported!')