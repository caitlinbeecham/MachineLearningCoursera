function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
%disp('m');
%disp(m);
n = size(X,2) - 1;
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
h = 1./(1.+exp(-(X*theta)));
%disp('size h should be m x 1');
%disp(size(h));
%disp('size y');
%disp(size(y));
%disp('size theta');
%disp(size(theta));
%disp('theta');
%disp(theta);
log_h = log(h);
one = ones(m,1);
ones_minus_h = one - h;
log_ones_minus_h = log(ones_minus_h);
%disp('size log h... should be same as size h');
%disp(size(log_h));
%disp('size one minus... should be same as size h');
%disp(size(ones_minus_h));
%disp('size log of one minus... should be same as size h');
%disp(size(log_ones_minus_h));
neg_y = -1*y;
y_minus_one = y - one;
len_theta = size(theta,1);
theta_short = theta(2:len_theta,1);
theta_short_sq = theta_short.^2;

J = ((1/m)*((neg_y)'*log_h + (y_minus_one)'*log_ones_minus_h)) + ((lambda/(2*m))*(sum(theta_short_sq)));
%disp('size J');
%disp(size(J));

%compute grad
offset = zeros(n+1,1);
for i = 2:n+1;
    offset(i,1) = (lambda/m)*theta(i);
end;
grad = (1/m)*(((h-y)'*X)') + offset;
%---------------------------------

% =============================================================

end
