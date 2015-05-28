function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size1, ...
                                   hidden_layer_size2, ...
                                   num_labels, ...
                                   X, Y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network

Theta1 = reshape(nn_params(1:hidden_layer_size1 * (input_layer_size + 1)), ...
                 hidden_layer_size1, (input_layer_size + 1));
 
first=1+hidden_layer_size1 * (input_layer_size + 1);
second=hidden_layer_size1 * (input_layer_size + 1)+hidden_layer_size2 * (hidden_layer_size1+ 1);


Theta2 = reshape(nn_params(first:second), ...
                 hidden_layer_size2, (hidden_layer_size1 + 1));
                 
first=1+hidden_layer_size1 * (input_layer_size + 1)+hidden_layer_size2 * (hidden_layer_size1+ 1);     

Theta3 = reshape(nn_params(first:end), ...
                 num_labels, (hidden_layer_size2 + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Theta3_grad = zeros(size(Theta3));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
%%完全向量化版
a1 = [ones(m,1) X]; %5000*401
z2 = a1*Theta1';  %5000*25
a2 = [ones(size(z2,1),1) sigmoid(z2)]; %5000*(25+1)
z3 = a2*Theta2'; %5000*10
a3=[ones(size(z3,1),1) sigmoid(z3)];
z4=a3*Theta3';
a4 = sigmoid(z4);
h=a4;
%-----------------Part 3: Compute Cost (Feedforward)-------   -------------
% Y=zeros(m,num_labels);
% for i=1:num_labels
%     Y(:,i)=(y==i);
% end
J = 1/m*sum(sum(-Y.*log(h)-(1-Y).*log(1-h)));
%--------------------------------------------------------------------------
%%正则化后
J = J + lambda/2/m*( sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2))+sum(sum(Theta3(:,2:end).^2)));
% compute delta
delta4=zeros(m, num_labels);
% for k = 1 : num_labels
%     delta3(:,k) = a3(:,k) - (y==k); %5000*10
% end
delta4=a4-Y;
delta3=delta4*Theta3.*[ones(size(z3,1),1) sigmoidGradient(z3)];
delta2 = delta3(:,2:end)*Theta2 .*[ones(size(z2,1),1) sigmoidGradient(z2)]; %5000*26

%compute Delta
Delta1 = delta2(:,2:end)' * a1;  %25*401
Delta2 = delta3(:,2:end)' * a2;
Delta3 = delta4' * a3;
% Delta2 = delta3' * a2; %10*26

% compute Theta_grad
Theta1_grad = 1/m*Delta1;
Theta2_grad = 1/m*Delta2;
Theta3_grad = 1/m*Delta3;
% 正则化grad
reg1 = lambda/m*Theta1;
reg2 = lambda/m*Theta2;
reg3 = lambda/m*Theta3;
reg1(:,1) = 0;
reg2(:,1) = 0;
reg3(:,1) = 0;
Theta1_grad = Theta1_grad + reg1;
Theta2_grad = Theta2_grad + reg2;
Theta3_grad = Theta3_grad + reg3;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:);Theta3_grad(:)];


end
