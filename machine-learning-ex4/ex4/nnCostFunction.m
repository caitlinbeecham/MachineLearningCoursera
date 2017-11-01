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
%X = [ones(m,1) X];

all_hs = zeros(m,num_labels);
all_ys = zeros(m,num_labels);
n = size(Theta1,1);
h = size(Theta2,2) - 1;
r = num_labels;
%for i = 1:10;
for i = 1:m;
    y_vec = zeros(1,num_labels);
    %disp("y");
    %disp(y(i));
    %disp("y_vec before");
    %disp(y_vec);
    y_vec(1,y(i)) = 1;
    %disp("y_vec after");
    %disp(y_vec);
    all_ys(i,:) = y_vec;
end;
%disp("y_1");
%disp(y(1,1));
%disp("y_vec 1")
%disp(y_vec(1,:));
Delta_1 = zeros(size(Theta1,1),size(Theta1,2));
Delta_2 = zeros(size(Theta2,1),size(Theta2,2));
all_a2s = zeros(m,(h+1));
all_z2s = zeros(m,h);
%all_delta3s = zeros();
%all_delta2s = zeros();

for i=1:m;
    %do part 1
    %do part 2
    %do part 3
    a_1 = X(i,:)';
    %disp("size a_1 1");
    %disp(size(a_1,1));
    %disp("size a_1 2");
    %disp(size(a_1,2));
    
    a_1 = [1;a_1];
    %{
disp("size a_1 1");
    disp(size(a_1,1));
    disp("size a_1 2");
    disp(size(a_1,2));
    disp("size Theta1 1");
    disp(size(Theta1,1));
    disp("size Theta1 2");
    disp(size(Theta1,2));
    disp("size Theta2 1");
    disp(size(Theta2,1));
    disp("size Theta2 2");
    disp(size(Theta2,2));
    %}
    z_2 = Theta1*a_1;
    %disp("size z_2");
    %disp(size(z_2));
    all_z2s(i,:) = z_2';
    a_2 = 1./(1.+exp(-(z_2)));
    %disp("size a_2 without bias unit");
    %disp(size(a_2));
    a_2 = [1;a_2];
    all_a2s(i,:)= a_2';
    %disp("size a_2 with bias unit");
    %disp(size(a_2));
    z_3 = Theta2*a_2;
    a_3 = 1./(1.+exp(-(z_3)));
    h_x = a_3;
    all_hs(i,:) = h_x';
    %{
    delta_3 = a_3 - y(i);
disp("size delta_3");
    disp(size(delta_3));
    disp("size theta2 with bias");
    disp(size(Theta2));
    disp("size of Theta 2 transpose");
    disp(size(Theta2'));
    disp("size of delta_3");
    disp(size(delta_3));
    disp("size of Theta2' * delta_3");
    disp(size(Theta2'*delta_3));
    disp("size of g'(z_2)");
    %}
    %disp(size(((exp(-z_2))./(1+exp(-z_2).^2))));
    %g_prime_without_bias = (exp(-z_2))./(1+exp(-z_2).^2);
    %g_prime_with_bias = [0;g_prime_without_bias];
    %if have bug change this 0 to 1
    %delta_2 = (Theta2'*delta_3).*(g_prime_with_bias);
    %{
    disp("size Delta 1");
    disp(size(Delta_1));
    disp("size delta 2 withput delta^2_0 removed");
    disp(size(delta_2));
    disp("size of a_1 transpose");
    disp(size(a_1'));
    disp("size of delta_2 without rem * a_1 transpose");
    disp(size(delta_2*(a_1')));
    
    %Delta_1 = Delta_1 + delta_2(2:end)*(a_1');
    %Delta_2 = Delta_2 + delta_3*(a_2');
%}
end;
%disp(size(y));
delta_3 = all_hs - all_ys;
%disp("size all z2s");
%disp(size(all_z2s));
delta_2 = delta_3*Theta2(:,2:end);
delta_2 = delta_2 .* sigmoidGradient(all_z2s);
%now all_hs(k,i) is what we want in the sum for h_theta(x^(i))_k
%as shown in the assignment paper
%disp("size delta_2");
%disp(size(delta_2));
%disp("size X");
%disp(size(X));
Delta1 = delta_2'*[ones(size(X,1),1) X];
%disp("d2");
%disp(delta_2);
%disp("d3");
%disp(delta_3);
%disp("Delta1");
%disp(Delta1);
%disp("size delta_3");
%disp(size(delta_3));
%disp("size all_a2s");
%disp(size(all_a2s));
Delta2 = delta_3'*all_a2s;
%disp("Delta2");
%disp(Delta2);
Theta1_grad = (1/m)*Delta1;
Theta2_grad = (1/m)*Delta2;
%disp("Theta1_grad");
%disp(Theta1_grad);
%disp("Theta2_grad");
%disp(Theta2_grad);
%a_1 = X(1,:);
%a_1 = [1;a_1];
%z_2 = Theta1*a_1;
%a_2 = 1./(1.+exp(-(z_2)));
%a_2 = [1;a_2];
%z_3 = Theta2*a_2;
%a_3 = 1./(1.+exp(-(z_3)));
%h_x = a_3;
%disp("k");
%disp(num_labels);
%disp("m");
%disp(m);
%disp("size y 1");
%disp(size(y,1));
%disp("size y 2");
%disp(size(y,2));
%disp("size all_hs 1");
%disp(size(all_hs,1));
%disp("size all_hs 2");
%disp(size(all_hs,2));



J = 0.0;
for i = 1:m;
    for k = 1:num_labels;
       J = J + (-all_ys(i,k)*log(all_hs(i,k)) - (1-all_ys(i,k))*(log(1-all_hs(i,k)))); 
    end;
end;
J = J/m;
reg_part_J = 0;
l_1 = size(Theta1,1);
l_2 = size(Theta1,2); 
%but remember loop for l_2 will start at 2
l_3 = size(Theta2,1);
l_4 = size(Theta2,2);
%but remember loop for l_4 will start at 2

%for i = 1:10;
 %   for j = 2:10;
        %disp("Theta1 i j squared");
        %disp(Theta1(i,j)^2);
   % end;
    
%end;
for i = 1:l_1;
    for j = 2:l_2;
        reg_part_J = reg_part_J + (Theta1(i,j))^2;
    end;
end;

for i = 1:l_3;
    for j = 2:l_4;
        reg_part_J = reg_part_J + (Theta2(i,j))^2;
    end;
end;
%disp("reg_part_J");
%disp(reg_part_J);
reg_part_J = reg_part_J*(lambda/(2*m));
%disp("reg_part_J");
%disp(reg_part_J);
J = J + reg_part_J;


for i = 1:size(Theta1_grad,1);
    for j = 2:size(Theta1_grad,2);
        Theta1_grad(i,j) = Theta1_grad(i,j) + (lambda/m)*Theta1(i,j);
    end;
end;

for i = 1:size(Theta2_grad,1);
    for j = 2:size(Theta2_grad,2);
        Theta2_grad(i,j) = Theta2_grad(i,j) + (lambda/m)*Theta2(i,j);
    end;
end;





% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
