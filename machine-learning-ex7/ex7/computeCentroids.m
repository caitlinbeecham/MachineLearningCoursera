function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);

%size(idx,1) = m;
% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%
for i=1:K;
    indices = find(idx==i);
    sum = zeros(1,n);
    %disp("size(sum)");
    %disp(size(sum));
    for j=1:size(indices,1);
        %disp("j");
        %disp(j);
        %disp("size(sum) within j loop before adding");
        %disp(size(sum));
        %disp("size(X(indices(j),:)) should be same as size(sum)");
        %disp(size(X(indices(j),:)));
        sum = sum + X(indices(j),:);
        %disp("size(sum) within j loop after adding");
        %disp(size(sum));
    end;
    %disp("size indices");
    %disp(size(indices));
    sum = sum *(1/size(indices,1));
    %disp("size(sum)");
    %disp(size(sum));
    centroids(i,:) = sum;
end;







% =============================================================


end

