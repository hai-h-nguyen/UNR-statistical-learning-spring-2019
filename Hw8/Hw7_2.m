%% Preprocess images
clc;
clear all;
close all;

batch_1 = load("cifar-10-batches-mat/data_batch_1.mat");
batch_2 = load("cifar-10-batches-mat/data_batch_2.mat");
batch_3 = load("cifar-10-batches-mat/data_batch_3.mat");
batch_4 = load("cifar-10-batches-mat/data_batch_4.mat");
batch_5 = load("cifar-10-batches-mat/data_batch_5.mat");

data_1 = batch_1.data;
label_1 = batch_1.labels;

data_2 = batch_2.data;
label_2 = batch_2.labels;

data_3 = batch_3.data;
label_3 = batch_3.labels;

data_4 = batch_4.data;
label_4 = batch_4.labels;

data_5 = batch_5.data;
label_5 = batch_5.labels;

%% Print five images from the dataset
subplot(2,3,1)
image_1 = zeros(32, 32, 3, 'uint8');
image_1(:, :, 1) = reshape(data_1(1, 1:1024), [32 32]);
image_1(:, :, 2) = reshape(data_1(1, 1025:2048), [32 32]);
image_1(:, :, 3) = reshape(data_1(1, 2049:3072), [32 32]);
imshow(image_1)

subplot(2,3,2)
image_1 = zeros(32, 32, 3, 'uint8');
image_1(:, :, 1) = reshape(data_2(1, 1:1024), [32 32]);
image_1(:, :, 2) = reshape(data_2(1, 1025:2048), [32 32]);
image_1(:, :, 3) = reshape(data_2(1, 2049:3072), [32 32]);
imshow(image_1)

subplot(2,3,3)
image_1 = zeros(32, 32, 3, 'uint8');
image_1(:, :, 1) = reshape(data_3(1, 1:1024), [32 32]);
image_1(:, :, 2) = reshape(data_3(1, 1025:2048), [32 32]);
image_1(:, :, 3) = reshape(data_3(1, 2049:3072), [32 32]);
imshow(image_1)

subplot(2,3,4)
image_1 = zeros(32, 32, 3, 'uint8');
image_1(:, :, 1) = reshape(data_4(1, 1:1024), [32 32]);
image_1(:, :, 2) = reshape(data_4(1, 1025:2048), [32 32]);
image_1(:, :, 3) = reshape(data_4(1, 2049:3072), [32 32]);
imshow(image_1)

subplot(2,3,5)
image_1 = zeros(32, 32, 3, 'uint8');
image_1(:, :, 1) = reshape(data_5(1, 1:1024), [32 32]);
image_1(:, :, 2) = reshape(data_5(1, 1025:2048), [32 32]);
image_1(:, :, 3) = reshape(data_5(1, 2049:3072), [32 32]);
imshow(image_1)

%% Compute the mean value for the ten classes and print them
mean_class = zeros(32, 32, 3);
count = 0;

total_data = vertcat(data_1, data_2, data_3, data_4, data_5);
total_label = vertcat(label_1, label_2, label_3, label_4, label_5);

for image = 1: size(total_data, 1)
    if (total_label(image) == 0)
        mean_class(:, :, 1) = mean_class(:, :, 1) + double(reshape(total_data(image, 1:1024), [32 32]));
        mean_class(:, :, 2) = mean_class(:, :, 2) + double(reshape(total_data(image, 1025:2048), [32 32]));
        mean_class(:, :, 3) = mean_class(:, :, 3) + double(reshape(total_data(image, 2049:3072), [32 32]));
        count = count + 1;
    end
end

    mean_class_1 = uint8(mean_class/count);
    mean_class = zeros(32, 32, 3);
    count = 0;

for image = 1: size(total_data, 1)    
    if (total_label(image) == 1)
        mean_class(:, :, 1) = mean_class(:, :, 1) + double(reshape(total_data(image, 1:1024), [32 32]));
        mean_class(:, :, 2) = mean_class(:, :, 2) + double(reshape(total_data(image, 1025:2048), [32 32]));
        mean_class(:, :, 3) = mean_class(:, :, 3) + double(reshape(total_data(image, 2049:3072), [32 32]));
        count = count + 1;
    end
end
    mean_class_2 = uint8(mean_class/count);
    mean_class = zeros(32, 32, 3);
    count = 0;    
    
for image = 1: size(total_data, 1)    
    if (total_label(image) == 2)
        mean_class(:, :, 1) = mean_class(:, :, 1) + double(reshape(total_data(image, 1:1024), [32 32]));
        mean_class(:, :, 2) = mean_class(:, :, 2) + double(reshape(total_data(image, 1025:2048), [32 32]));
        mean_class(:, :, 3) = mean_class(:, :, 3) + double(reshape(total_data(image, 2049:3072), [32 32]));
        count = count + 1;
    end
end
    mean_class_3 = uint8(mean_class/count);
    mean_class = zeros(32, 32, 3);
    count = 0;        
    
for image = 1: size(total_data, 1)    
    if (total_label(image) == 3)
        mean_class(:, :, 1) = mean_class(:, :, 1) + double(reshape(total_data(image, 1:1024), [32 32]));
        mean_class(:, :, 2) = mean_class(:, :, 2) + double(reshape(total_data(image, 1025:2048), [32 32]));
        mean_class(:, :, 3) = mean_class(:, :, 3) + double(reshape(total_data(image, 2049:3072), [32 32]));
        count = count + 1;
    end
end
    mean_class_4 = uint8(mean_class/count);
    mean_class = zeros(32, 32, 3);
    count = 0;    
    
for image = 1: size(total_data, 1)    
    if (total_label(image) == 4)
        mean_class(:, :, 1) = mean_class(:, :, 1) + double(reshape(total_data(image, 1:1024), [32 32]));
        mean_class(:, :, 2) = mean_class(:, :, 2) + double(reshape(total_data(image, 1025:2048), [32 32]));
        mean_class(:, :, 3) = mean_class(:, :, 3) + double(reshape(total_data(image, 2049:3072), [32 32]));
        count = count + 1;
    end
end
    mean_class_5 = uint8(mean_class/count);
    mean_class = zeros(32, 32, 3);
    count = 0;    
    
for image = 1: size(total_data, 1)    
    if (total_label(image) == 5)
        mean_class(:, :, 1) = mean_class(:, :, 1) + double(reshape(total_data(image, 1:1024), [32 32]));
        mean_class(:, :, 2) = mean_class(:, :, 2) + double(reshape(total_data(image, 1025:2048), [32 32]));
        mean_class(:, :, 3) = mean_class(:, :, 3) + double(reshape(total_data(image, 2049:3072), [32 32]));
        count = count + 1;
    end
end
    mean_class_6 = uint8(mean_class/count);
    mean_class = zeros(32, 32, 3);
    count = 0;    
   
for image = 1: size(total_data, 1)
    if (total_label(image) == 6)
        mean_class(:, :, 1) = mean_class(:, :, 1) + double(reshape(total_data(image, 1:1024), [32 32]));
        mean_class(:, :, 2) = mean_class(:, :, 2) + double(reshape(total_data(image, 1025:2048), [32 32]));
        mean_class(:, :, 3) = mean_class(:, :, 3) + double(reshape(total_data(image, 2049:3072), [32 32]));
        count = count + 1;
    end
end
    mean_class_7 = uint8(mean_class/count);
    mean_class = zeros(32, 32, 3);
    count = 0;    

for image = 1: size(total_data, 1)
    if (total_label(image) == 7)
        mean_class(:, :, 1) = mean_class(:, :, 1) + double(reshape(total_data(image, 1:1024), [32 32]));
        mean_class(:, :, 2) = mean_class(:, :, 2) + double(reshape(total_data(image, 1025:2048), [32 32]));
        mean_class(:, :, 3) = mean_class(:, :, 3) + double(reshape(total_data(image, 2049:3072), [32 32]));
        count = count + 1;
    end
end
    mean_class_8 = uint8(mean_class/count);
    mean_class = zeros(32, 32, 3);
    count = 0;    
  
for image = 1: size(total_data, 1)
    if (total_label(image) == 8)
        mean_class(:, :, 1) = mean_class(:, :, 1) + double(reshape(total_data(image, 1:1024), [32 32]));
        mean_class(:, :, 2) = mean_class(:, :, 2) + double(reshape(total_data(image, 1025:2048), [32 32]));
        mean_class(:, :, 3) = mean_class(:, :, 3) + double(reshape(total_data(image, 2049:3072), [32 32]));
        count = count + 1;
    end
end
    mean_class_9 = uint8(mean_class/count);
    mean_class = zeros(32, 32, 3);
    count = 0;    

for image = 1: size(total_data, 1)
    if (total_label(image) == 9)
        mean_class(:, :, 1) = mean_class(:, :, 1) + double(reshape(total_data(image, 1:1024), [32 32]));
        mean_class(:, :, 2) = mean_class(:, :, 2) + double(reshape(total_data(image, 1025:2048), [32 32]));
        mean_class(:, :, 3) = mean_class(:, :, 3) + double(reshape(total_data(image, 2049:3072), [32 32]));
        count = count + 1;
    end
end
    mean_class_10 = uint8(mean_class/count);
    mean_class = zeros(32, 32, 3);
    count = 0;    

figure
subplot(2,5,1)
imshow(mean_class_1);

subplot(2,5,2)
imshow(mean_class_2);

subplot(2,5,3)
imshow(mean_class_3);

subplot(2,5,4)
imshow(mean_class_4);

subplot(2,5,5)
imshow(mean_class_5);

subplot(2,5,6)
imshow(mean_class_6);

subplot(2,5,7)
imshow(mean_class_7);

subplot(2,5,8)
imshow(mean_class_8);

subplot(2,5,9)
imshow(mean_class_9);

subplot(2,5,10)
imshow(mean_class_10);


