function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;
count = 0;
stepsize = (max(pval) - min(pval)) / 1000;
%disp("size(pval,1) == size(yval,1)");
%disp(size(pval,1) == size(yval,1));
%disp("pval");
%disp(pval);
%disp("stepsize");
%disp(stepsize);
%disp("min(pval)");
%disp(min(pval));
%disp("max(pval)");
%disp(max(pval));
%disp(pval);
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions
    predictions = (pval < epsilon);
    %disp("yval");
    %disp(yval);
    %disp("predictions");
    %disp(predictions);
    %the ones are the predicted anomalies
    %compute f1 score
    %tp = 0;
    %fp = 0;
    %fn = 0;
    %for j=1:size(yval,1);
    %    if yval(j) == 1;
    %        if predictions(j) == 1;
    %            tp = tp + 1;
    %        else;
    %            fn = fn + 1;
    %        end; 
    %    elseif predictions(j) == 1;
    %        fp = fp + 1;
    %    end;
    %end;
    %disp("tp");
    %disp(tp);
    %disp("fp");
    %disp(fp);
    %disp("fn");
    %disp(fn);
    tp = sum((predictions)&(yval));
    fp = sum((predictions)&(yval==0));
    fn = sum((predictions==0)&(yval));
    prec = tp/(tp+fp);
    rec = tp/(tp+fn);
    F1 = (2 * prec * rec)/(prec + rec);
    %disp("F1");
    %disp(F1);
    %disp("tp");
    %disp(tp);
    %disp("fp");
    %disp(fp);
    %disp("fn");
    %disp(fn);
  
    %disp("F1");
    %disp(F1);
    %disp("count");
    %disp(count);
    count = count + 1;
    % =============================================================
    %disp("F1");
    %disp(F1);
    %disp("bestF1");
    %disp(bestF1);
    if F1 > bestF1;
       %disp("f1 was better than best f1");
       %disp("------------");
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
