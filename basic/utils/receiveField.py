# [filter size, stride, padding]
# Assume the two dimensions are the same
# Each kernel requires the following parameters:
# - k_i: kernel size
# - s_i: stride
# - p_i: padding (if padding is uneven, right padding will higher than left padding; "SAME" option in tensorflow)
#
# Each layer i requires the following parameters to be fully represented:
# - n_i: number of feature (data layer has n_1 = imagesize )
# - j_i: distance (projected to image pixel distance) between center of two adjacent features
# - r_i: receptive field of a feature in layer i
# - start_i: position of the first feature's receptive field in layer i (idx start from 0, negative means the center fall into padding)

import math

convnet = [[11, 4, 0], [3, 2, 0], [5, 1, 2], [3, 2, 0], [3, 1, 1], [3, 1, 1], [3, 1, 1], [3, 2, 0], [6, 1, 0],
           [1, 1, 0]]
layer_names = ['conv1', 'pool1', 'conv2', 'pool2', 'conv3', 'conv4', 'conv5', 'pool5', 'fc6-conv', 'fc7-conv']
imsize = 227

class LayerInfo:
    def __init__(self, layer_name,kernel_size, stride, padding):
        self.layer_name= layer_name
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.jump=0
        self.receiveField =0

    def __repr__(self):
        ret = "{} : kernel_size: {}, stride: {}, padding:{}, jump: {}, receiveField: {}".format(
            self.layer_name, self.kernel_size, self.stride, self.padding, self.jump, self.receiveField
        )
        return ret

class LayerInfoList:
    def __init__(self, configs, names):
        self.layerInfos =list()
        for item in zip(configs, names):
            assert(len(item[0]) == 3)
            self.layerInfos.append(LayerInfo(item[1],item[0][0], item[0][1], item[0][2]))

    def computeReceiveField(self, imageSize):
        for index, item in enumerate(self.layerInfos):
            if(index == 0) :
                n_in = imageSize
                j_in = 1
                r_in = 1
            else:
                n_in = self.layerInfos[index-1].kernel_size
                j_in = self.layerInfos[index-1].jump
                r_in = self.layerInfos[index-1].receiveField

            k = item.kernel_size
            s = item.stride
            p = item.padding

            n_out = math.floor((n_in - k + 2 * p) / s) + 1
            actualP = (n_out - 1) * s - n_in + k
            pR = math.ceil(actualP / 2)
            pL = math.floor(actualP / 2)
            j_out = j_in * s
            r_out = r_in + (k - 1) * j_in

            item.jump = j_out
            item.receiveField = r_out

    def show(self):
        for item in self.layerInfos:
            print(item)

if __name__ == "__main__":
    convnet1 =[[5,1,0],[2,2,0],[3,1,0],[2,2,0],[3,1,0]]
    names1 = ['conv1','pool1','conv2','pool2','conv3']
    layers = LayerInfoList(convnet1, names1)
    layers.computeReceiveField(48)
    layers.show()
    print('......')

    convnet2 = [[3,2,1],[1,1,0],[3,1,1],[2,2,0],[1,1,0],[3,1,1],[2,2,0],[1,1,0],[3,1,1],[1,1,1], [6,6,0]]
    names2 =['conv1','conv2_1','conv2_3','pool2','conv3_1','conv3_3','pool3','conv4_1','conv4_3','conv5','avgpool']
    layers = LayerInfoList(convnet2, names2)
    layers.computeReceiveField(48)
    layers.show()