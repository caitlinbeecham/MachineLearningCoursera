function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
for i =1:size(X,1);
    min_square_dist_from_centroid = inf;
    closest_centroid_idx = -1;
    for j=1:size(centroids,1);
        %ok so size (x,2) = size(centroids,2)
        component_dist_from_current_centroid = centroids(j,:)-X(i,:);
        square_component_dist_from_current_centroid = component_dist_from_current_centroid.^2;
        square_dist_from_current_centroid = sum(square_component_dist_from_current_centroid);
        if square_dist_from_current_centroid < min_square_dist_from_centroid;
            min_square_dist_from_centroid = square_dist_from_current_centroid;
            closest_centroid_idx = j;
        end;
    end;
    idx(i) = closest_centroid_idx;
end;



% =============================================================

end

