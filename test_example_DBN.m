function test_example_DBN
%%说明：
%0-这个程序的功能，使用DBN算法来实现人脸识别，数据库使用ORL数据库，在迭代次数达到3000时，识别准确率98%左右
%1-此程序使用的是LBP（Local Binary Pattern）来实现特征的提取
%2-此程序还附带了画学习曲线的功能，画正则参数，隐层结点数，训练样本数的学习曲线
%3-DBN是有几层RBM构成，我这个程序实现的是4层网络，输入层-隐层1-隐层2-输出层
%4-DBN的训练基本上分为两步，先用RBM的训练方法训练网络得到初始值，来初始化整个网络，然后用BP反向传播算法来微调整个网络
%5-神经网络参数的更新，使用matlab提供的fmincg函数，前提是我们要先得到网络的代价函数nnCostFunction
%%
%--------------------------对ORL数据库进行LBP特征提取------------------------
mapping=getmapping(8,'u2');%先计算Lbp算子的映射表

train_x=[];%%训练数据集
for i=1:40%----------------%ORL数据库共有40人-------------------------------
    for j=1:7%-------------%每个人选择7张来进行训练-------------------------
        a=imread(strcat('E:\My RBM-DBN matlab\ORL\ORL\s',num2str(i),'_',num2str(j),'.bmp'));
        c=a;
        row=size(c,1);%读入图片，并对图片进行分块，采用4*4分块，每块进行LBP
        col=size(c,2);
        B=mat2cell(c,[row/4 row/4 row/4 row/4],[col/4 col/4 col/4 col/4]);
        H.a=0;        %对每个子块进行Lbp
        for k=1:16
        H1=lbp(B{k},1,8,mapping,'h'); %LBP histogram in (8,1) neighborhood %using uniform patterns
        H.hist{k}=H1;
        end   
        hist=[H.hist{1},H.hist{2},H.hist{3},H.hist{4},H.hist{5},H.hist{6},H.hist{7},H.hist{8},H.hist{9},H.hist{10},H.hist{11},H.hist{12},H.hist{13},H.hist{14},H.hist{15},H.hist{16}];
        MappedData = mapminmax(hist, 0, 0.5);%将输入数据归一化到[0,0.5]
        train_x=[train_x;MappedData];
    end
end
train_y=zeros(280,40);%%训练样本，输出标签,40*7=280
for i=1:40
    for j=1:7
        train_y((i-1)*7+j,i)=1;
    end
end
%-----------------------------------------------------------------------------
test_x=[];%%测试数据集
for i=1:40%----------------%ORL数据库共有40人-------------------------------
    for j=8:10%------------%每个人选择3张来进行训练-------------------------
        %------------------%E:\My RBM-DBN matlab\ORL\ORL\s，数据库的路径----
        a=imread(strcat('E:\My RBM-DBN matlab\ORL\ORL\s',num2str(i),'_',num2str(j),'.bmp'));
        c=a;
        row=size(c,1);%读入图片，并对图片进行分块，2 
        col=size(c,2);
        B=mat2cell(c,[row/4 row/4 row/4 row/4],[col/4 col/4 col/4 col/4]);
        H.a=0;                       %对每个子块进行Lbp
        for k=1:16
        H1=lbp(B{k},1,8,mapping,'h'); %LBP histogram in (8,1) neighborhood %using uniform patterns
        H.hist{k}=H1;
        end   
        hist=[H.hist{1},H.hist{2},H.hist{3},H.hist{4},H.hist{5},H.hist{6},H.hist{7},H.hist{8},H.hist{9},H.hist{10},H.hist{11},H.hist{12},H.hist{13},H.hist{14},H.hist{15},H.hist{16}];
        MappedData = mapminmax(hist, 0, 0.5);
        test_x=[test_x;MappedData];
    end
end
test_y=zeros(120,40);%%训练标签
for i=1:40
    for j=1:3
        test_y((i-1)*3+j,i)=1;
    end
end
%--------数据转换成double型----------------------------------------------------
train_x = double(train_x);
test_x  = double(test_x) ;
train_y = double(train_y);
test_y  = double(test_y);
%%  ex1 train a 100 hidden unit RBM and visualize its weights
% rand('state',0)
% dbn.sizes = [100];
% % opts.numepochs =   1;
% opts.numepochs =5;
% % opts.batchsize = 100;
% opts.batchsize = 1;
% opts.momentum  =   0;
% opts.alpha     =   1;
% dbn = dbnsetup(dbn, train_x, opts);
% dbn = dbntrain(dbn, train_x, opts);
% figure; visualize(dbn.rbm{1}.W');   %  Visualize the RBM weights

%%  ex2 train a 100-100 hidden unit DBN and use its weights to initialize a NN
%---------------设置DBN网络的各种参数---------------------------------------
rand('state',0)
%train dbn
dbn.sizes = [100 100];%DBN的两个隐层是100-100
opts.numepochs = 30;   %迭代次数为30
opts.batchsize = 1;   %每次处理batchsize个数据
opts.momentum  =   0;
opts.alpha     =    0.001;%学习率为0.01
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);
%%
%--------------------DBN网络参数初始化NN-----------------------------------
input_layer_size  = size(train_x,2);  %输入数据的维数，对应可见层的数目
hidden_layer_size1=100;
hidden_layer_size2=100;
num_labels=40;                        %所以，整个4层网络是input_layer_size-100-100-40
% Theta1=randInitializeWeights(input_layer_size,hidden_layer_size1);
% Theta2=randInitializeWeights(hidden_layer_size1,hidden_layer_size2);
Theta1=[dbn.rbm{1}.c dbn.rbm{1}.W];   %训练好的DBN参数来初始化神经网络
Theta2=[dbn.rbm{2}.c dbn.rbm{2}.W];   %
Theta3=randInitializeWeights(hidden_layer_size2,num_labels);%最后输出层用随机初始化
initial_nn_params = [Theta1(:) ; Theta2(:);Theta3(:)];
lambda = 0.00003;%正则化参数抑制过拟合
%%
%--------------------------训练神经网络-------------------------------------
nn_params=train_nn(initial_nn_params,lambda,train_x,train_y,...
          input_layer_size,hidden_layer_size1, hidden_layer_size2,num_labels);
%%
%------------------------对训练完后的NN进行预测和性能测试--------------------
%将参数矩阵还原
Theta1 = reshape(nn_params(1:hidden_layer_size1 * (input_layer_size + 1)), ...
                 hidden_layer_size1, (input_layer_size + 1));
 
first=1+hidden_layer_size1 * (input_layer_size + 1);
second=hidden_layer_size1 * (input_layer_size + 1)+hidden_layer_size2 * (hidden_layer_size1+ 1);
Theta2 = reshape(nn_params(first:second), ...
                 hidden_layer_size2, (hidden_layer_size1 + 1));
                 
first=1+hidden_layer_size1 * (input_layer_size + 1)+hidden_layer_size2 * (hidden_layer_size1+ 1);     
Theta3 = reshape(nn_params(first:end), ...
                 num_labels, (hidden_layer_size2 + 1));

%进行预测
pred = predict(Theta1, Theta2,Theta3, test_x);
%计算正确率
[dummy, expected] = max(test_y,[],2);
 bad = find(pred ~= expected);    
 er = numel(bad) / size(test_x, 1);
 
 assert(er < 0.10, 'Too big error');
%%
%--------------描绘关于隐层结点数的学习曲线----------------------------------
% lambda = 0.03;%正则化参数
% [hidden_node, error_train, error_val] =hidden_node_learn_curve(lambda,train_x,train_y,test_x,test_y);
% figure(2);
% plot(hidden_node, error_train, hidden_node, error_val);
% legend('Train', 'Cross Validation');
% xlabel('hidden node');
% ylabel('Error');
% axis([100 200 0 1])
%--------------------------------------------------------------------------
%%
%------------------描绘关于训练样本规模的学习曲线----------------------------
% % lambda =1;%正则化参数
% lambda = 0.03;%正则化参数
% figure(1);
% % m=size(train_x,1)/20; %m=20,描20个点
% m=3:7;
% [error_train, error_val] = ...
%     learningCurve(initial_nn_params,train_x,train_y,test_x,test_y, lambda,...
%     input_layer_size,hidden_layer_size1,hidden_layer_size2,num_labels);
% plot(m, error_train, m, error_val);
% 
% title(sprintf('Learning Curve (lambda = %f)', lambda));
% xlabel('Number of training examples')
% ylabel('Error')
% axis([3 7 0 5])
% legend('Train', 'Cross Validation')

%--------------------------------------------------------------------------
%%
%---------------描绘关于正则化参数lambda的学习曲线---------------------------
% [lambda_vec, error_train, error_val] = ...
%     validationCurve(initial_nn_params,train_x,train_y,test_x,test_y,...
%     input_layer_size,hidden_layer_size1,hidden_layer_size2,num_labels);
% 
% close all;
% figure(2);
% plot(lambda_vec, error_train, lambda_vec, error_val);
% legend('Train', 'Cross Validation');
% xlabel('lambda');
% ylabel('Error');
% axis([0 0.04 0 10])
%---------------------------------------------------------------------------
