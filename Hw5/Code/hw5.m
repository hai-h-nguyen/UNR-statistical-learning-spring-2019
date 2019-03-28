clc
clear all
close all

% Import data
M = importdata("prostate.data");
N_sample = length(M);

% Exclude the first row (labels only)
for i = 2 : N_sample - 1
    temp = cell2mat(M(i, :));
    temp = strsplit(temp);
    
    % Exclude index column (column 1), True/False column (last column)
    if i == 2
        data = str2double(temp(1, 2:end-1));  
    else
        data = [data; str2double(temp(1, 2:end-1))];
    end
end

% Total number of predictors
p = 8;

%% Best subset selection
% Fit all models
for k = 1:p
    total_models = combnk(1:p, k);
    
    [total_combination, width] = size(total_models);
    
    % Output
    y = data(:, end);
    
    max_Rsq = 0.0;
    best_feature = 9;
    
    for i = 1 : total_combination     
        % Construct vector X
        chosen_features = total_models(i, :);
        X = data(:, chosen_features);
        X = [ones(length(X),1) X];
        
        % Coefficient
        b = inv(X'*X) * X' * y;
        
        % Estimated output
        y_hat = X * b;
                
        % Calculate R^2
        Rsq = 1 - sum((y - y_hat).^2)/sum((y - mean(y)).^2);
        
        if Rsq > max_Rsq
            max_Rsq = Rsq;
            best_feature = chosen_features;
        end     
    end
    
    % Store best models with k predictors
    Model{k} = [max_Rsq best_feature];
end

% Select a single best model among saved models using adjusted R-squared
for index = 1:p
    % Retrieve R squared and the number of predictors used
    r_squared = Model{index}(1);
    n_predictor = length(Model{index}) - 1;
    
    % Calculate adjusted R squared
    adj_rsquared = 1 - (1-r_squared) * (N_sample - 1) / (N_sample - n_predictor - 1);
    
    % For plotting
    adj_R_squared(index) = adj_rsquared;
end

% Draw figure of adjusted R squared and the number of predictors
y_axis = 1:1:p;
plot(y_axis, adj_R_squared, 'LineWidth', 2);
xlabel('Number of Predictors');
ylabel('Adjusted R^2');
grid on;

%% Ridge regression
% Analytical solution
% Normalize predictor
X_ridge = data(:, 1:end-1);
for col = 1:p
    mean_ = mean(X_ridge(:, col));
    std_  = std(X_ridge(:, col));
    X_ridge(:, col) = (X_ridge(:, col) - mean_)./std_;
end

y_ridge = data(:, end);
lambda = 0:1:10^4;

for index = 1:length(lambda)
    temp = inv(X_ridge' * X_ridge + lambda(index) * eye(p)) * X_ridge' * y_ridge;
    if index == 1
        b_ridge = temp;
    else
        b_ridge = [b_ridge temp];
    end
end

% Plot
semilogx(lambda, b_ridge, 'LineWidth', 2)
xlabel('Lambda')
ylabel('Standardized Coefficient')
legend('lcavol','lweight', 'age', 'lbph','svi','lcp','gleason', 'pgg45')
grid on

%% Lasso regression
X_lasso = [ones(size(data,1),1) X_ridge];
y_lasso = data(:, end);
s_array = [0.01 0.05 0.1 0.5 1 5 10];
b_lasso = [];
for i = 1:length(s_array)
    beta = zeros(size(X_lasso,2),1);
    [beta,costHistory] = grad_descent_lasso(X_lasso, y_lasso, beta, 0.1, 100, s_array(i));
    if index == 1
        b_lasso = beta;
    else
        b_lasso = [b_lasso beta];
    end   
end
figure
semilogx(s_array, b_lasso, 'LineWidth', 2)
xlabel('s')
ylabel('Standardized Coefficient')
legend('beta 0','lcavol','lweight', 'age', 'lbph','svi','lcp','gleason', 'pgg45')
grid on

