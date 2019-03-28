function [parameters,costHistory] = grad_descent_lasso(X, y, parameters, default_learningRate, num_iter, s)

m = length(y);

% Creating a matrix of zeros for storing our cost function history
costHistory = zeros(num_iter, 1); 

for i = 1:num_iter        

    % Calculating the transpose of our hypothesis
    h = (X * parameters - y);        

    % Updating the parameters
    tmp_parameters = parameters - default_learningRate * (1/m) * X' * h;
    while (norm(tmp_parameters, 1) > s)
        default_learningRate = default_learningRate * 0.5;
        tmp_parameters = parameters - default_learningRate * (1/m) * X' * h;        
    end
    parameters = tmp_parameters;
    % Keeping track of the cost function
    costHistory(i) = 1/m*(X * parameters - y)'*(X * parameters - y);        
end 
end


