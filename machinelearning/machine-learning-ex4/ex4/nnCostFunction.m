function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
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
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

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

X = [ones(m, 1) X];
n = size(X, 2) - 1;
K1 = size(Theta1,1);
K2 = size(Theta2, 1); 

% size:   X:  m*(n+1);      
%    Theta1:  K1*(n+1);
%    Theta2:  K2*(K1+1);
%         y:  m*1
%        y2:  m*K2
y2 = zeros(m, K2);
for i=1:m
	row = zeros(1, K2);
	c = y(i);
	row(c) = 1;
   y2(i,:) = row;
end

%  a1 = X;              m*(n+1)
%  a2 = g(a1*Theta1');    m*K1
%  a2 = [ones(m, 1) a2]; m*(K1+1)
%  a3 = g(a2*Theta2');  m*K2 
%  y2 :  m * K2

a1 = X;
z2 = a1 * Theta1';  %  m*(n+1) by (n+1)*K1;  m*K1
a2 = sigmoid(a1 * Theta1');
a2 = [ones(m, 1) a2];
a3 = sigmoid(a2 * Theta2');
hx = a3;
J1 =  -y2 .* log(hx) - (1 - y2) .* log( 1 - hx);
J2 = sum(J1, 2);
J3 = sum(J2);

pTheta1 = Theta1(:, 2:(n+1));
pTheta2 = Theta2(:, 2:(K1+1));
J4 = sum(sumsq(pTheta1)) + sum(sumsq(pTheta2));

J = ( J3 / m) + lambda*(J4/(2*m));


err3 = a3 - y2;  % m*K2
err2 = (err3 * pTheta2) .* ...   % m*K2*K1
      sigmoidGradient(z2);     % m*K1

%size(pTheta1)
%size(pTheta2)

tmp_theta2 = lambda / m * Theta2;
tmp_theta2(:,1) = 0;

tmp_theta1 = lambda / m * Theta1;
tmp_theta1(:, 1) = 0;

delta2 = zeros(size(Theta2)); % K2*K1
delta2a = err3' * a2(:,1:end) / m;%   %  m*K2   m*K1
delta2b = err3' * a2(:,1:end) / m +  tmp_theta2;%   %  m*K2   m*K1
%size(delta2)

delta1 = zeros(size(Theta1)); % K1 * n
delta1a = err2' * a1(:, 1:end) / m;%  + lambda ./ Theta1;   % m*K1  m*n
delta1b = err2' * a1(:, 1:end) / m  + tmp_theta1;%  + lambda ./ Theta1;   % m*K1  m*n
%size(delta1)

Theta1_grad = delta1b;%[delta1a(1,:); delta1b(2:end,:)];
Theta2_grad = delta2b;%[delta2a(2,:); delta2b(2:end,:)];









% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
