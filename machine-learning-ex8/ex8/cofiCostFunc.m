function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%
%disp("num_movies, should be equal to size 1 X");
%disp(num_movies);
%disp("num_users should be the second size dim of Y");
%disp(num_users);
%disp("size(R)");
%disp(size(R));
%disp("size(Y)");
%disp(size(Y));
%disp("size(X)");
%disp(size(X));
%disp(" size X* Theta'   should be same as Y");
%disp(size(X*Theta'));
%disp("size(Theta) should be num users by num features");
%disp(size(Theta));
%disp(" ");
%disp(" ");
%disp(" ");
%for i = 1:num_movies;
%    for j = 1:num_users;
%        if R(i,j) == 1;
%            disp("size of theta(j)'   size 1 should be num features, size 2 should be 1");
%            disp(size((Theta(j,:))'));
%            disp("size of X(i,:) should be 1");
%            disp(size(X(i,:)));
%            J = J + (((X(i,:)*(Theta(j,:))')-Y(i,j))^2);
%        end;
%    end;
%end;
%J = J/2;
J = sum(sum((.5)*((((X*Theta') - Y).*R).^2)));
J = J + (lambda/2)*(sum(sum(Theta.^2)) + sum(sum(X.^2)));
%disp("size(Theta)");
%disp(size(Theta));
%disp("size(X*Theta')");
%disp(size(X*Theta'));
%disp("size Y");
%disp(size(Y));
%disp("size((((X*Theta')-Y).*R)))");
%disp(size((((X*Theta')-Y).*R)));
X_grad = (((X*Theta')-Y).*R)*Theta + lambda*X;
Theta_grad = (((X*Theta')-Y).*R)'*X + lambda*Theta;


% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
