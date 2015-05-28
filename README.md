# LBP-DBN-face-recognition
使用LBP特征提取算法提取人脸特征，DBN网络来实现人脸识别，测试数据库-ORL数据库，识别率可达90%以上<br>
###工程使用方法：<br>
clone下工程，要根据实际情况修改test_example_DBN.m中的文件路径<br>
我们在该文件中，a=imread(strcat('E:\My RBM-DBN matlab\ORL\ORL\s',num2str(i),'_',num2str(j),'.bmp'));<br>
###工程说明：<br>
####英文简称注释：<br>
DBN-深度信念网络、RBM-受限的玻尔兹曼机 、LBP-局部二值模式<br>
####几点说明：<br>
0-这个程序的功能，使用DBN算法来实现人脸识别，数据库使用ORL数据库，在迭代次数达到3000时，识别准确率98%左右<br>
2-此程序还附带了画学习曲线的功能，画正则参数，隐层结点数，训练样本数的学习曲线<br>
3-DBN是有几层RBM构成，我这个程序实现的是4层网络，输入层-隐层1-隐层2-输出层<br>
4-DBN的训练基本上分为两步，先用RBM的训练方法训练网络得到初始值，来初始化整个网络，然后用BP反向传播算法来微调整个网络<br>
5-神经网络参数的更新，使用matlab提供的fmincg函数，前提是我们要先得到网络的代价函数nnCostFunction<br>

####各文件的功能说明：<br>
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
###优点：<br>
   神经网络采用的是随机初始化，但容易收敛到局部最小值，用DBN训练得到网络的初始化参数，而不是随机初始化。DBN随机初始化之后，网络的训练方式就跟训练神经网络一样。<br>

###想弄明白这个程序：需要了解LBP,RBM,DBN,BP神经网络的知识<br>
现在，把我实现这个工程，用到的各种资源贴上来，方便学习<br>
我的工程是在别人的代码上改进的，代码见此博客<br>
http://blog.csdn.net/dark_scope/article/details/9447967<br>

####LBP方向：<br>
[LBP算法的研究及其在人脸识别中的应用 ](http://blog.csdn.net/dujian996099665/article/details/9038303)<br>
[matlab学习：人脸识别之LBP (Local Binary Pattern)](http://www.cnblogs.com/yingying0907/archive/2012/11/18/2773920.html)<br>
[LBP算法的Matlab代码 ](http://blog.csdn.net/kuaitoukid/article/details/8643253)<br>
[opencv学习之（三）-LBP算法的研究及其实现 ](http://blog.csdn.net/dujian996099665/article/details/8886576)<br>
[LBP原理加源码解析  ](http://blog.csdn.net/xidianzhimeng/article/details/19634573)<br>
[LBP特征学习及实现  ](http://blog.csdn.net/jinshengtao/article/details/18219697)<br>
[目标检测的图像特征提取之（一）HOG特征](http://blog.csdn.net/liulina603/article/details/8291093)<br>
[目标检测的图像特征提取之（二）LBP特征  ](http://blog.csdn.net/liulina603/article/details/8291105)<br>

####RBM，DBN相关的资料：<br>
[rbm C++代码理解  ](http://blog.csdn.net/u012878523/article/details/39179101)<br>
[【面向代码】学习 Deep Learning（二）Deep Belief Nets(DBNs)  ](http://blog.csdn.net/dark_scope/article/details/9447967)<br>
[DeepLearning（深度学习）原理与实现（三） ](http://blog.csdn.net/marvin521/article/details/8896636)<br>
[Deep Belief Network(DBN)的实现(c++) ](http://www.zhizhihu.com/html/y2013/4365.html)<br>
[yusugomori/DeepLearning ](https://github.com/yusugomori/DeepLearning)<br>
[DeepLearnToolbox DBN源码解析  ](http://blog.csdn.net/chlele0105/article/details/20781985)<br>

####BP神经网络：<br>
[神经网络-UFLDL](http://deeplearning.stanford.edu/wiki/index.php/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C)<br>

####视频资源，强烈推荐看一下couresa上吴恩达的视频machine learing ，可以看一下这几节课<br>
1、	IX: Neural Networks: Learning (Week 5)<br> 介绍反向传播算法，梯度校验（梯度校验可以检查我们的梯度下降法是否正确，也可以校验我们的反向传播算法是否正确）<br>
2、	VII：正则化，我的程序中使用了正则参数<br>
3、	X：Advice for Applying Machine Learning (Week 6)：讲述过拟合，欠拟合之类的,并教授描绘学习曲线的方法。<br>
