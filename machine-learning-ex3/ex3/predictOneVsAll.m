function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
%disp("m");
%disp(m);
num_labels = size(all_theta, 1);
%disp("num_labels");
%disp(num_labels);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%   
%disp("size all_theta ... should be num_labels x (n+1)");
%disp(size(all_theta));
%disp("size X transpose ... should be (n+1) x m");
%disp(size(X'));
result = all_theta*X';
%disp("size result before transposing ... should be num_labels x m");
%disp(size(result));
%for i=1:m;
 %   result(:,i) = 1./(1.+exp(-(result(:,i))));
%end;
answer = zeros(size(result,1),size(result,2));

for i=1:num_labels;
    for j=1:m;
        answer(i,j) = 1/(1+exp(-(result(i,j))));
    end;
end;
%result = result';
%disp("size result after transposing ... should be m x num_labels");
%disp(size(result));
%disp(result(1,:));
answer = answer';
[max_elts, p] = max(answer, [], 2);
%disp("size p ... should be m x 1");
%disp(size(p));
%disp("size answer .. should be same as size result");
%disp(size(answer));
%disp("answer");
%disp(answer(floor(size(answer,1)*.25):floor(size(answer,1)*.75)));
%disp("p");
%disp(p(floor(size(answer,1)*.25):floor(size(answer,1)*.75)));







% =========================================================================


end
