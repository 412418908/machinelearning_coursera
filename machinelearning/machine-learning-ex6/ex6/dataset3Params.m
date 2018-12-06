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

C_vec = [0.01,0.03,0.1,0.3,1,3]';
sigma_vec = C_vec;
trys = length(C_vec);
err_vec = zeros(trys, trys);

for i=1:trys
	for j=1:trys
		C = C_vec(i);
		sigma = sigma_vec(j);
		model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
		pred = svmPredict(model, Xval);
		
		err_vec(i, j) = mean(double(pred ~= yval));
	end
	
end

%{
printf("[i,j]\tErr\tC\tSigma\t\n");
for i=1:trys
	for j=1:trys
		C = C_vec(i);
		sigma = sigma_vec(j);
		err = err_vec(i, j);
		printf("[%d,%d]\t%f\t%f\t%f\n", i,j,err,C,sigma);
	end
	
end
%}

[colMin, j] = min(min(err_vec)); 
[minValue, i] = min(err_vec(:,j));
C = C_vec(i); sigma = sigma_vec(j);






% =========================================================================

end
