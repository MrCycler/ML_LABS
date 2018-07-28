function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
thetatemp = theta;
thetatemp(1) = 0;
% You need to return the following variables correctly 
J = 0;
J = J + (sum((X*theta - y).^2)/(2*m))+ ((lambda/(2*m))*(sum(thetatemp.^2))); 
grad = zeros(size(theta));
grad(1) = (sum((X*theta-y).*X(:,1))/m);
for i=2:size(theta)
grad(i)= (sum((X*theta-y).*X(:,i))/m)+((lambda/m)*theta(i));
end
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% =========================================================================

grad = grad(:);

end
