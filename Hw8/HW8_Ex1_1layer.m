close all
clear all

% read, display 1 image
fileID_test_X = fopen('t10k-images.idx3-ubyte');
test_data = fread(fileID_test_X);

fileID_test_Y = fopen('t10k-labels.idx1-ubyte');
test_label = fread(fileID_test_Y);

fileID_training_X = fopen('train-images.idx3-ubyte');
training_data = fread(fileID_training_X);

fileID_training_Y = fopen('train-labels.idx1-ubyte');
training_label = fread(fileID_training_Y);

% TODO: calculate from data 
training_num_img = 60000;
test_num_img = 10000;
img_cols = 28;
img_rows = 28;

% training params
input_layer_size  = img_rows * img_cols;
hidden_layer1_size = 256;
output_layer_size = 10;

lr0 = 0.001;
decay_rate = 1;%0.95;

N_epoch = 300;

minibatch_size = 1000;
dropout_keeprate_input = 0.8;
dropout_keeprate_hidden_1 = 0.5;

%% process data
training_data_X = reshape(training_data(17:end),[img_cols*img_rows training_num_img])';
training_data_Y = zeros(training_num_img, 10);
for i=1:training_num_img
    training_data_Y(i, training_label(8+i)+1) = 1;
end
% Randomly shuffle samples (rows)
shuffledRowIdx = randperm(training_num_img);
training_data_X = training_data_X(shuffledRowIdx, :);
training_data_Y = training_data_Y(shuffledRowIdx, :);

test_data_X = reshape(test_data(17:end),[img_cols*img_rows test_num_img])';
test_data_Y_digit = test_label(9:end);
test_data_Y = zeros(test_num_img, 10);
for i=1:test_num_img
    test_data_Y(i, test_label(8+i)+1) = 1;
end

%% Initialization
o_hat = horzcat(training_data_X, ones(training_num_img, 1));

% Xavier Initialization
W1_ = normrnd(0, sqrt(2/(input_layer_size + hidden_layer1_size)),...
    [input_layer_size + 1, hidden_layer1_size]);
W2_ = normrnd(0, sqrt(2/(hidden_layer1_size + output_layer_size)),...
    [hidden_layer1_size + 1, output_layer_size]);

%% Training
start_idx = 1;
end_idx = start_idx + minibatch_size - 1;
for epoch = 1 : N_epoch
    d_W2_  = zeros(size(W2_));
    d_W1_  = zeros(size(W1_));
    dropout_mask_in_layer = [(rand([1,input_layer_size]) <= dropout_keeprate_input) 1];
    dropout_mask_hidden_layer1 = [(rand([1,hidden_layer1_size]) <= dropout_keeprate_hidden_1) 1];
    % One episode run through all sample
    %% todo: minibatch
    for i = start_idx:end_idx % not so correct
        
        W2  = W2_(1 : end - 1, :);
        W1  = W1_(1 : end - 1, :);
        
        % Feed-forward
        o_hat(i,:) = o_hat(i,:) .* dropout_mask_in_layer;
        out_1 = o_hat(i, :) * W1_;
        o_1 = sigmoid(out_1);
        o_1_hat = horzcat(o_1, ones(size(o_1, 1), 1));
        
        o_1_hat = o_1_hat .* dropout_mask_hidden_layer1;
        out_2 = o_1_hat * W2_;
        o_2 = sigmoid(out_2);
        
        % Back-prop
        e = o_2 - training_data_Y(i,:);
        
        v2 = o_2 .* (1.-o_2);
        D2 = diag(v2);
        
        v1 = o_1 .* (1.-o_1);
        D1 = diag(v1);
        
        delta_2 = D2 * e';
        delta_1 = D1 * W2 * delta_2;
        
        % Decay learning rate
        lr = lr0 * decay_rate^(epoch - 1);
        
        % Weight correction
        d_W2_t = -lr * delta_2 * o_1_hat;
        d_W1_t = -lr * delta_1 * o_hat(i, :);
        
        % Add correction vector
        d_W2_ =  d_W2_ + d_W2_t';
        d_W1_ =  d_W1_ + d_W1_t';
    end
    
    % Update weights
    W2_ = W2_ + d_W2_;
    W1_ = W1_ + d_W1_;
    
    % Feed-forward
    t_o_hat = horzcat(test_data_X, ones(test_num_img, 1));
    W1_compensated = [W1_(1:end-1,:) * dropout_keeprate_input; W1_(end,:)];
    t_out_1 = t_o_hat * W1_compensated;
    t_o_1 = sigmoid(t_out_1);
    t_o_1_hat = horzcat(t_o_1, ones(size(t_o_1, 1), 1));
    
    W2_compensated = [W2_(1:end-1,:) * dropout_keeprate_hidden_1; W2_(end,:)];
    t_out_2 = t_o_1_hat * W2_compensated;
    t_o_2 = sigmoid(t_out_2);
    
    [tmp, digit_out]=max(t_o_2,[],2);
    digit_out = digit_out - 1;
    error_rate(epoch) = sum(test_data_Y_digit ~= digit_out)/test_num_img;
    
    start_idx = mod(start_idx + minibatch_size, training_num_img);
    end_idx = mod(start_idx + minibatch_size - 1, training_num_img);  
end

%% Plot
plot(error_rate);
grid on;