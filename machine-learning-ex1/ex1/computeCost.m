function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

predictions_vector = X * theta;
%disp('');
%disp('');
%disp('size theta..... should be (n+1) x 1');
%disp(size(theta));
%disp('');
%disp('');

%disp('');
%disp('');
%disp('size predictions_vector.....should be same as y');
%disp(size(predictions_vector));
%disp('');
%disp('');
%disp('');
%disp('');
%disp('size y');
%disp(size(y));
%disp('');
%disp('');
diff_vector = predictions_vector - y;
%disp('');
%disp('');
%disp('size diff_vector..... should be (m) x 1');
%disp(size(diff_vector));
%disp('');
%disp('');
square_diff_vector = diff_vector .^ 2;

%disp('');
%disp('');
%disp('size square_diff_vector..... should be (m) x 1');
%disp(size(square_diff_vector));
%disp('');
%disp('');
sum_square_diff = sum(square_diff_vector);

%disp('');
%disp('');
%disp('sum_square_diff');
%disp(sum_square_diff);
%disp(size(sum_square_diff));
%disp('');
%disp('');
J = (1/(2*m)) * sum_square_diff;

%disp('');
%disp('');
%disp('size of J');
%disp(size(J));
%disp('');
%disp('');
%disp('');






% =========================================================================

end
