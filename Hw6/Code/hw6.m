close all
clear all
clc

% Number of samples of each class
Sample = 10;

% Define inputs and outputs
offset = 5; % offset for second class
x_org = [randn(2,Sample) randn(2,Sample)+offset]; 
y_org = [zeros(1,Sample) ones(1,Sample)];       
%% Perceptron learning
% Extended vector
x_ext = vertcat(x_org, ones(1,2*Sample));

% % Start: The weight vector w_0 is generated randomly,
w = rand(1,3);

% Test: Select randomly a vector x from P U N
while 1  
    % Check stop condition: stop when all vectors are classified correctly
    y_est_all = w * x_ext;
    y_est_all = y_est_all > 0;
    if isequal(y_est_all, logical(y_org))
        break;
    end
    
    random_int = randi(length(x_ext));
    x_rand = x_ext(:, random_int);

    y_est = w * x_rand;
    y_true = y_org(random_int);    

    if (y_true == 1) && (y_est <= 0)
        w = w + x_rand';
    end

    if (y_true == 0) && (y_est > 0)
        w = w - x_rand';
    end
        
end

plotpv(x_org,y_org);
plotpc(w(:,1:end-1),w(end));
grid on
%% Logistic Regression
x = x_org';
y = y_org';
[m, n] = size(x); 

x = [ones(m, 1) x];

initial_theta = zeros(n + 1, 1);
iterations = 2500;
alpha = 0.025;

[theta, J] = gradientDescent(x, y, initial_theta, alpha, iterations);

% Plot Boundary
plotDecisionBoundary(theta, x, y);
grid on

