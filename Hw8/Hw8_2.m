%% Preprocess images
clc;
clear all;
close all;

for i = 1:5
    batch(i) = load(strcat("cifar-10-batches-mat/data_batch_", int2str(i), ".mat"));
end

%% Print five images from the dataset
for i = 1:5
    subplot(2,3,i)
    image_idx = zeros(32, 32, 3, 'uint8');
    image_idx(:, :, 1) = reshape(batch(i).data(1, 1:1024), [32 32]);
    image_idx(:, :, 2) = reshape(batch(i).data(1, 1025:2048), [32 32]);
    image_idx(:, :, 3) = reshape(batch(i).data(1, 2049:3072), [32 32]);
    imshow(imrotate(image_idx, -90))
end

%% Compute the mean value for the ten classes and print them
mean_class = zeros(32, 32, 3);
count = 0;

total_data = batch(1).data;
total_label = batch(1).labels;
for i = 2:5
    total_data = vertcat(total_data, batch(i).data);
    total_label = vertcat(total_label, batch(i).labels);
end

mean_class = zeros(32, 32, 3, 10);

for class_id = 1:10
    for image_idx = 1: size(total_data, 1)
        if (total_label(image_idx) == class_id - 1)
            mean_class(:, :, 1, class_id) = mean_class(:, :, 1, class_id) + double(reshape(total_data(image_idx, 1:1024), [32 32]));
            mean_class(:, :, 2, class_id) = mean_class(:, :, 2, class_id) + double(reshape(total_data(image_idx, 1025:2048), [32 32]));
            mean_class(:, :, 3, class_id) = mean_class(:, :, 3, class_id) + double(reshape(total_data(image_idx, 2049:3072), [32 32]));
            count = count + 1;
        end
    end
    
    mean_class(:, :, :, class_id) = uint8(mean_class(:, :, :, class_id)/count);
    count = 0;    
end

figure
for class_id = 1:10
    subplot(2,5,class_id)
    imshow(uint8(mean_class(:, :, :, class_id)));
    title("Class " + int2str(class_id - 1));
end