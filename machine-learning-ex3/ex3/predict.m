function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
col_ones = ones(m,1);
X = [col_ones X];
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
all_probs = zeros(m,num_labels);
for i=1:m;
    current_x = (X(i,:))';
    theta_1_x = Theta1*current_x;
    g_of = 1./(1.+exp(-(theta_1_x)));
    g_of = [1; g_of];
    theta_2_g_of = Theta2*g_of;
    g_of_2 = 1./(1.+exp(-(theta_2_g_of)));
    g_of_2 = g_of_2';
    all_probs(i,:) = g_of_2;
end;
%disp("size p(1)");
%disp(size(p(1)));
[max_num, p] = max(all_probs, [], 2);








% =========================================================================


end
