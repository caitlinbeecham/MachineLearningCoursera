function [U, S] = pca(X)
%PCA Run principal component analysis on the dataset X
%   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
%

% Useful values
[m, n] = size(X);

% You need to return the following variables correctly.
U = zeros(n);
S = zeros(n);

% ====================== YOUR CODE HERE ======================
% Instructions: You should first compute the covariance matrix. Then, you
%               should use the "svd" function to compute the eigenvectors
%               and eigenvalues of the covariance matrix. 
%
% Note: When computing the covariance matrix, remember to divide by m (the
%       number of examples).
%

%covariance matrix is (1/m)*sum_i=1^m (x(i)'*x(i))
%it might be beacuse its actually the ith row, wait year it's the ith row
%wait no it's the ith column?
sum_matrix = 0;
for i=1:m;
    sum_matrix = sum_matrix + X(i,:)'*X(i,:);
end;
cov_matrix = (1/m)*sum_matrix;

[U, S, V] = svd(cov_matrix);


% =========================================================================

end