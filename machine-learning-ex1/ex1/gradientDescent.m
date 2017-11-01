function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
%disp('m');
%disp(m);
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    
    % predictions_vector is an (i x 1) vector containing h_theta(x^(i)) at
    % the ith index
    %disp('size of theta at start of gradient descent');
    %disp(size(theta));
    %disp('');
    %disp('size of X should be m x (n+1) = (97) x (2)');
    %disp(size(X));
    predictions_vector = X * theta;
    %disp('size of predictions vector... should be same as y');
    %disp(size(predictions_vector));
    %disp('size of y');
    %disp(size(y));
    diff_vector = predictions_vector - y;
    %disp('size of diff vector and size of X');
    %disp(size(diff_vector));
    %disp(size(X));
    %disp('size of diff vector transpose');
    %disp(size((diff_vector)'));
    delta = (1/m)*(X' * (diff_vector));
    %disp('size of delta... should be (n+1) x 1... n+1 = 2');
    %disp(size(delta));
    %disp('size of alpha * delta');
    %disp(size(alpha * delta));
    %disp('size of theta before update, should be same as above');
    %disp(size(theta));
    theta = theta - (alpha * delta);
    %disp('size of theta after update');
    %disp(size(theta));
    %disp('');
    
    
    cost = computeCost(X,y,theta);
    %disp('cost and size of cost');
    %disp(cost);
    %disp(size(cost));
    %disp('');




    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
