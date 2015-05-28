function [lambda_vec, error_train, error_val] = ...
    validationCurve(initial_nn_params,train_x,train_y,test_x,test_y,...
    input_layer_size,hidden_layer_size1,hidden_layer_size2,num_labels)

% lambda_vec =0.5:0.02:1.5;
%lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10];
lambda_vec=0:0.02:0.5;
error_train = zeros(size(lambda_vec,2), 1);
error_val = zeros(size(lambda_vec,2), 1);
for i=1:length(lambda_vec)
    lambda=lambda_vec(i);
    theta=train_nn(initial_nn_params,lambda,train_x,train_y,...
                  input_layer_size,hidden_layer_size1, hidden_layer_size2,num_labels);
    [error_train(i),grad]=nnCostFunction(theta,input_layer_size,hidden_layer_size1,...
                         hidden_layer_size2,num_labels,train_x,train_y,0); 
    [error_val(i),  grad]=nnCostFunction(theta,input_layer_size,hidden_layer_size1,...
                         hidden_layer_size2,num_labels,test_x,test_y,0); 
end

end

