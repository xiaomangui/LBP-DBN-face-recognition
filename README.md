# LBP-DBN-face-recognition
使用LBP特征提取算法提取人脸特征，DBN网络来实现人脸识别，测试数据库-ORL数据库，识别率可达90%以上<br>
工程说明：<br>
英文简称注释：<br>
DBN-深度信念网络、RBM-受限的玻尔兹曼机 、LBP-局部二值模式<br>
几点说明：<br>
0-这个程序的功能，使用DBN算法来实现人脸识别，数据库使用ORL数据库，在迭代次数达到3000时，识别准确率98%左右<br>
2-此程序还附带了画学习曲线的功能，画正则参数，隐层结点数，训练样本数的学习曲线<br>
3-DBN是有几层RBM构成，我这个程序实现的是4层网络，输入层-隐层1-隐层2-输出层<br>
4-DBN的训练基本上分为两步，先用RBM的训练方法训练网络得到初始值，来初始化整个网络，然后用BP反向传播算法来微调整个网络<br>
5-神经网络参数的更新，使用matlab提供的fmincg函数，前提是我们要先得到网络的代价函数nnCostFunction<br>

各文件的功能说明：<br>
test_example_DBN：主函数<br>
dbnsetup：初始化DBN网络<br>
dbntrain:训练DBN网络，DBN是由多层RBM组成，训练方法是逐层训练，先训练第一层网络，然后固定第一层网络的参数，将第一层网络的输出作为下一层网络的输入<br>
dbnunfoldtonn：DBN训练得到的参数来初始化神经网络<br>
fmincg：最优化函数，只要我们得到网络的代价函数，和反向传播算法，就可以用此函数求最优解<br>
getmapping，lbp,lbptest:完成lbp算法<br>
hidden_node_learn_curve:关于隐层结点的学习曲线<br>
learningCurve：关于训练样本数目的学习曲线<br>
nnCostFunction：求神经网络的代价函数，和BP反向传播算法<br>
predict：利用网络进行预测<br>
randInitializeWeights：随机初始化网络参数<br>
rbmdown，rbmup，sigm，sigmoid，sigmoidGradient，sigmrnd：训练过程中使用到的计算函数<br>
train_nn：训练神经网络<br>
rbmtrain：训练一个RBM网络<br>
validationCurve：关于正则参数的学习曲线<br>

整个框架就是：<br>
    先用LBP算子对人脸图像进行特征提取，然后用RBM算法逐层训练RBM，训练得到的RBM参数来初始化化DBN网络，然后计算出整个网络的代价函数，并使用BP反向传播算法，谨记反向传播算法也是梯度下降算法，并且可以用梯度校验来验证。梯度下降算法，学习率是一个比较关键的参数，但是，在我的程序中，自己不设学习率，只要我们实现了反向传播算法和代价函数，我们使用fmincg函数去得到最优解。<br>
优点：<br>
   神经网络采用的是随机初始化，但容易收敛到局部最小值，用DBN训练得到网络的初始化参数，而不是随机初始化。DBN随机初始化之后，网络的训练方式就跟训练神经网络一样。<br>


