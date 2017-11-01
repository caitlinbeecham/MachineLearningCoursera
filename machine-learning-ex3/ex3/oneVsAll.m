function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
%disp("m");
%disp(m);
n = size(X, 2);
%disp("size y");
%disp(size(y));
%disp("y");
%disp(y);

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%

%for each class ie set of elt in num_labels with same vals
%eg 10, 1 2 3  4 5 6 7 8 9
%want to make set of training example indices with that val
%and set without
%then want to run logistic regression to find boundary between those two
%groups
%to find the column vector for each label 1,2,3,4,5,6,7,8,9,10 use
%arrays_for_each_label(:,label)
%that will be the right column vector

%now make vectors of 0's and 1's for each of ten classes
%each of these vectors will be 1 if the y value is equal to the value of
%each class
%can make this as one matrix
% will have m rows and 10 cols

helpful_matrix = zeros(m,num_labels);
for i=1:num_labels;
    helpful_matrix(:,i) = (y == i);
end;
%disp(helpful_matrix);

for c=1:num_labels;
    %train the logistic classifier for class i
    %then store it in the ith row of all_theta
    current_y = helpful_matrix(:,i);
    initial_theta = zeros(n + 1, 1);

    options = optimset('GradObj', 'on', 'MaxIter', 50);

    [theta] = ...
         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
                 initial_theta, options);
    all_theta(c,:) = theta;
end;









% =========================================================================


end
