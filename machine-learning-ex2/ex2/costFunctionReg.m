function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

tempTheta = theta(2:end,:);

pos = find(y==1); neg = find(y == 0);
hypothesisMatrix = sigmoid(X*theta);
%hypothesisMatrix = hypothesisMatrix(2:end, : );
J = ((sum(-log(hypothesisMatrix(pos)))+ sum(-log(1-hypothesisMatrix(neg))))/m) + lambda*sum(tempTheta.^2)/(2*m);

grad = (sum((hypothesisMatrix - y).*X)/m)';
grad += ([0; tempTheta]*lambda )/m;

% =============================================================

end
