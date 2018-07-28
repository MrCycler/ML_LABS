function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

J= (sum((-y.*log(sigmoid(X*theta)))-(-y+1).*log(1.-sigmoid(X*theta)))/m)+(lambda*(theta'*theta)/(2*m))-lambda*theta(1)*theta(1)/(2*m);

grad(1)=((sum((sigmoid(X*theta)-y)'*X(:,1)))/m);
for iter2=2:n
grad(iter2)=((sum((sigmoid(X*theta)-y)'*X(:,iter2)))/m)+lambda*theta(iter2)/m;
    % ============================================================
end


% =============================================================

end
