function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

m = length(y); 
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
  saved_theta = theta;  
  theta = theta - alpha*1/m*X'*(sigmoid(X*saved_theta)-y);  
  J_history(iter) = 1/m*sum(-y'*log(sigmoid(X*theta))-(1.-y)'*log(1-sigmoid(X*theta)));
end

end
