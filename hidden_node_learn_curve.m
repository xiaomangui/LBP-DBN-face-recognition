%描绘关于隐层结点的学习曲线
function [hidden_node, error_train, error_val] =hidden_node_learn_curve(lambda,train_x,train_y,test_x,test_y)
    hidden_node=50:10:200;
    error_train = zeros(size( hidden_node,2), 1);
    error_val = zeros(size(hidden_node,2), 1);
    for i=1:length(hidden_node)
    node=hidden_node(i);
    %训练两层的dbn网络
    rand('state',0)
    dbn.sizes = [node node];
    opts.numepochs = 1;
    opts.batchsize = 1;
    opts.momentum  =   0;
    opts.alpha     =    0.01;
    dbn = dbnsetup(dbn, train_x, opts);
    dbn = dbntrain(dbn, train_x, opts);
    input_layer_size  = size(train_x,2);  % 20x20 Input Images of Digits
    hidden_layer_size1=node;
    hidden_layer_size2=node;
    num_labels=40;

    Theta1=[dbn.rbm{1}.c dbn.rbm{1}.W];
    Theta2=[dbn.rbm{2}.c dbn.rbm{2}.W];
    Theta3=randInitializeWeights(hidden_layer_size2,num_labels);
    initial_nn_params = [Theta1(:) ; Theta2(:);Theta3(:)];
    
    
    theta=train_nn(initial_nn_params,lambda,train_x,train_y,...
                  input_layer_size,hidden_layer_size1, hidden_layer_size2,num_labels);
    [error_train(i),grad]=nnCostFunction(theta,input_layer_size,hidden_layer_size1,...
                         hidden_layer_size2,num_labels,train_x,train_y,0); 
    [error_val(i),  grad]=nnCostFunction(theta,input_layer_size,hidden_layer_size1,...
                         hidden_layer_size2,num_labels,test_x,test_y,0); 
    end
     
end
