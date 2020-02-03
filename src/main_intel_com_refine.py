
import numpy
import gc
import matplotlib
import matplotlib.pyplot as plt
from net.scalenet import ScaleNetParams, ScaleNet
from trainer.scalenettrainer import ScaleNetTrainerParams, ScaleNetTrainer
from util.handdetector import HandDetector
import os
import pickle
from data.importers import IntelImporter
from data.dataset import IntelDataset
from util.handpose_evaluation import IntelHandposeEvaluation
from util.helpers import shuffle_many_inplace

if __name__ == '__main__':

    eval_prefix = 'INTEL_COM_AUGMENT'
    if not os.path.exists('./eval/'+eval_prefix+'/'):
        os.makedirs('./eval/'+eval_prefix+'/')

    rng = numpy.random.RandomState(23455)

    print("create data")
    aug_modes = ['com', 'rot', 'none']  # 'sc',

    di = IntelImporter('../data/intel/')
    Seq2 = di.loadSequence('test_seq', docom=True)
    testSeqs = [Seq2]

    # Create testing data
    testDataSet = IntelDataset(testSeqs)
    test_data, test_gt3D = testDataSet.imgStackDepthOnly('test_seq')

    val_data = test_data
    val_gt3D = test_gt3D

    ####################################
    # resize data
    # dsize = (int(train_data.shape[2]//2), int(train_data.shape[3]//2))
    # xstart = int(train_data.shape[2]/2-dsize[0]/2)
    # xend = xstart + dsize[0]
    # ystart = int(train_data.shape[3]/2-dsize[1]/2)
    # yend = ystart + dsize[1]
    # val_data2 = val_data[:, :, ystart:yend, xstart:xend]

    # dsize = (int(train_data.shape[2]//4), int(train_data.shape[3]//4))
    # xstart = int(train_data.shape[2]/2-dsize[0]/2)
    # xend = xstart + dsize[0]
    # ystart = int(train_data.shape[3]/2-dsize[1]/2)
    # yend = ystart + dsize[1]
    # val_data4 = val_data[:, :, ystart:yend, xstart:xend]

    # dsize = (int(train_data.shape[2]//2), int(train_data.shape[3]//2))
    # xstart = int(train_data.shape[2]/2-dsize[0]/2)
    # xend = xstart + dsize[0]
    # ystart = int(train_data.shape[3]/2-dsize[1]/2)
    # yend = ystart + dsize[1]
    # test_data2 = test_data[:, :, ystart:yend, xstart:xend]

    # dsize = (int(train_data.shape[2]//4), int(train_data.shape[3]//4))
    # xstart = int(train_data.shape[2]/2-dsize[0]/2)
    # xend = xstart + dsize[0]
    # ystart = int(train_data.shape[3]/2-dsize[1]/2)
    # yend = ystart + dsize[1]
    # test_data4 = test_data[:, :, ystart:yend, xstart:xend]

    print((test_gt3D.max(), test_gt3D.min()))
    print((test_data.max(), test_data.min()))

    imgSizeW = test_data.shape[3]
    imgSizeH = test_data.shape[2]
    nChannels = test_data.shape[1]

    #############################################################################
    print("create network")
    batchSize = 64
    poseNetParams = ScaleNetParams(type=1, nChan=nChannels, wIn=imgSizeW, hIn=imgSizeH, batchSize=batchSize,
                                   resizeFactor=2, numJoints=1, nDims=3)
    poseNet = ScaleNet(rng, cfgParams=poseNetParams)

    # poseNetTrainerParams = ScaleNetTrainerParams()
    # poseNetTrainerParams.use_early_stopping = False
    # poseNetTrainerParams.batch_size = batchSize
    # poseNetTrainerParams.learning_rate = 0.0005
    # poseNetTrainerParams.weightreg_factor = 0.0001
    # poseNetTrainerParams.force_macrobatch_reload = True
    # poseNetTrainerParams.para_augment = True
    # poseNetTrainerParams.augment_fun_params = {'fun': 'augment_poses', 'args': {'normZeroOne': False,
    #                                                                             'di': di,
    #                                                                             'aug_modes': aug_modes,
    #                                                                             'hd': HandDetector(train_data[0, 0].copy(), abs(di.fx), abs(di.fy), importer=di)}}

    # print("setup trainer")
    # poseNetTrainer = ScaleNetTrainer(poseNet, poseNetTrainerParams, rng, './eval/'+eval_prefix)
    # poseNetTrainer.setData(train_data, train_gt3D[:, di.crop_joint_idx, :], val_data, val_gt3D[:, di.crop_joint_idx, :])
    # poseNetTrainer.addStaticData({'val_data_x1': val_data2, 'val_data_x2': val_data4})
    # poseNetTrainer.addManagedData({'train_data_x1': train_data2, 'train_data_x2': train_data4})
    # poseNetTrainer.addManagedData({'train_data_com': train_data_com,
    #                                'train_data_cube': train_data_cube,
    #                                'train_data_M': train_data_M,
    #                                'train_gt3D': train_gt3D})
    # poseNetTrainer.compileFunctions()

    ###################################################################
    # TRAIN
    # train_res = poseNetTrainer.train(n_epochs=100)
    # train_costs = train_res[0]
    # val_errs = train_res[2]

    # # plot cost
    # fig = plt.figure()
    # plt.semilogy(train_costs)
    # plt.show(block=False)
    # fig.savefig('./eval/'+eval_prefix+'/'+eval_prefix+'_cost.png')

    # fig = plt.figure()
    # plt.semilogy(val_errs)
    # plt.show(block=False)
    # fig.savefig('./eval/'+eval_prefix+'/'+eval_prefix+'_errs.png')

    # # save results
    # poseNet.save("./eval/{}/net_{}.pkl".format(eval_prefix, eval_prefix))
    poseNet.load("./eval/{}/net_{}.pkl".format(eval_prefix,eval_prefix))

    ####################################################
    # TEST
    print("Testing ...")
    gt3D = [j.gt3Dorig[di.crop_joint_idx].reshape(1, 3) for j in testSeqs[0].data]
    jts = poseNet.computeOutput([test_data])
    joints = []
    for i in range(test_data.shape[0]):
        joints.append(jts[i].reshape(1, 3)*(testSeqs[0].config['cube'][2]/2.) + testSeqs[0].data[i].com)

    hpe = IntelHandposeEvaluation(gt3D, joints)
    hpe.subfolder += '/'+eval_prefix+'/'
    print(("Mean error: {}mm, max error: {}mm".format(hpe.getMeanError(), hpe.getMaxError())))

    # save results
    pickle.dump(joints, open("./eval/{}/result_{}_{}.pkl".format(eval_prefix,os.path.split(__file__)[1],eval_prefix), "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    print("Testing baseline")

    #################################
    # BASELINE
    # Load the evaluation
    # data_baseline = di.loadBaseline('../data/ICVL/LRF_Results_seq_1.txt')

    # hpe_base = ICVLHandposeEvaluation(gt3D, numpy.asarray(data_baseline)[:, di.crop_joint_idx, :].reshape((len(gt3D), 1, 3)))
    # hpe_base.subfolder += '/'+eval_prefix+'/'
    # print(("Mean error: {}mm".format(hpe_base.getMeanError())))

    # com = [j.com for j in testSeqs[0].data]
    # hpe_com = ICVLHandposeEvaluation(gt3D, numpy.asarray(com).reshape((len(gt3D), 1, 3)))
    # hpe_com.subfolder += '/'+eval_prefix+'/'
    # print(("Mean error: {}mm".format(hpe_com.getMeanError())))