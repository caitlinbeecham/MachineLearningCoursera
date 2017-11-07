function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
c_list = [0.01,0.03,0.1,0.3,1.0,3,10,30];
sigma_list = [0.01,0.03,0.1,0.3,1.0,3,10,30];
best_c = 0.01;
best_sigma = 0.01;
min_cross_val_error = inf;
for c = c_list;
    for sig = sigma_list;
        model= svmTrain(X, y, c, @(x1, x2) gaussianKernel(x1, x2, sig));
        predictions = svmPredict(model,Xval);
        %disp("size predictions");
        %disp(size(predictions));
        %disp("predictions");
        %disp(predictions);
        % measure cross validation error
        %disp("size yval");
        %disp(size(yval));
        %disp("yval");
        %disp(yval);
        error_measure_vec = (yval == predictions);
        current_error = mean(double(predictions ~= yval));
        if current_error < min_cross_val_error;
            min_cross_val_error = current_error;
            best_c = c;
            best_sigma = sig;
        end;
    end;
end;
%disp("min_cross_val_error");

C = best_c;
sigma = best_sigma;



% =========================================================================

end
