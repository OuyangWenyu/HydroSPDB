import refine
# import argparse
import os
import imp
imp.reload(refine)
refine.reload()

if __name__ == '__main__':
    opt = refine.classLSTM.optLSTM()
    parser = opt.toParser()
    args = parser.parse_args()
    opt.fromParser(parser)
    print(opt)
    refine.funLSTM.trainLSTM(opt)


def screen(*, opt, cudaID, screenName='test'):
    argStr = opt.toCmdLine()
    codePath = os.path.realpath(__file__)
    cmd = 'CUDA_VISIBLE_DEVICES='+str(cudaID)+' ' + \
        'screen -dmS '+screenName+' '+'python '+codePath+argStr
    print(cmd)
    os.system(cmd)
