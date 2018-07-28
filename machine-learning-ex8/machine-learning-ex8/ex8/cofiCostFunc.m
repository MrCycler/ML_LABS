function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%Compute the cost:
costo=X*(Theta');
 [momo, nono] = size(R);
for i=1:momo
    for j=1:nono
        if(R(i,j)==1)
          J=J+((costo(i,j)-Y(i,j))*(costo(i,j)-Y(i,j))/2);
        end
    end
end
%Regularized form:
for i=1:momo
    J=J+lambda*(sum(X(i,:).*X(i,:)))/2;
end
for j=1:nono
    J=J+lambda*(sum(Theta(j,:).*Theta(j,:)))/2;
end


for i=1:momo
  idx = find(R(i, :)==1);
  Thetatemp = Theta(idx, :);
  Ytemp = Y(i, idx);
  X_grad(i, :) = ((X(i, :) *(Thetatemp')) -Ytemp)*Thetatemp+lambda*X(i, :);
end
for j=1:nono
  jdq = find(R(:, j)==1);
  Xtemp =X(jdq,:);
  Ytemp2 = Y(jdq, j);
  Theta_grad(j,:) = ((Xtemp*( Theta(j,:)')) -Ytemp2)'*Xtemp+lambda*Theta(j, :);
end
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%
















% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
