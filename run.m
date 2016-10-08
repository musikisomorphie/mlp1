clear; clc; 
%num of training set
par.tr_num = 278;
%num of testing set
par.te_num = 138;


%image height
im.h = 176;
%image width
im.w = 208;
%image channel
im.c = 176;
%image downscale rate
im.s = 0.1;

%load labels
label = csvread('targets.csv');
label = reshape(single(label), [1 par.tr_num]);

dt_size = round(im.h * im.s) * round(im.w * im.s) * im.c; 
train = zeros(dt_size, par.tr_num);
%% dimension reduction for training data
for i = 1:par.tr_num
    disp(num2str(i));
    nii = load_nii(['set_train/train_' num2str(i) '.nii']);
    image = single(nii.img);
    image = imresize(image, im.s);
    train(:, i) = reshape(image, [size(image, 1) * size(image, 2)* size(image, 3) 1]);
end
%squeeze out zeros
train(~any(train, 2), :) = [];
%extract training feature
[eig_vec, eig_val] = Eigenface_f(train,par.tr_num);

test = zeros(dt_size, par.te_num);
%% same operations as training data
for i = 1:par.te_num
    disp(num2str(i));
    nii = load_nii(['set_test/test_' num2str(i) '.nii']);
    image = single(nii.img);
    image = imresize(image, im.s);
    test(:, i) = reshape(image, [size(image, 1) * size(image, 2)* size(image, 3) 1]);
end
test(~any(test, 2), :) = [];


%feature dimension
dim = 250;
% normalize features
tr_feats = normalize_data((eig_vec(:,1:dim)'*train)','l2')';
te_feats = normalize_data((eig_vec(:,1:dim)'*test)','l2')';

% Least-squares estimation
beta = (tr_feats * tr_feats') \ (tr_feats * label');

%% test phase
fid=fopen('sol.csv','wt');
fprintf(fid,'%s\n', 'ID,Prediction');
for i = 1 : par.te_num
    disp(num2str(i));        
    y = te_feats(:,i)'* beta;
    y(y < 18) = 18;
    y(y > 90) = 90;
    fprintf(fid,'%d%s%d\n', i, ',', y);
end

fclose(fid);