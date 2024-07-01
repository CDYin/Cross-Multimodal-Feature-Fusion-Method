%% 最终分类网络结构
clear;clc;
% 加载数据
data1 = xlsread('.\data\featureData\fc-CBAM\fc-acoustic\train.xlsx');
data2 = xlsread('.\data\featureData\fc-CBAM\fc-acoustic\test.xlsx');
signalTrain=data1;
signalValidation = data2;

[row_Train, column_Train] = size(signalTrain);
[row_Validation, column_Validation] = size(signalValidation);
% 提取训练集训练数据和目标类别
P_train = signalTrain( : , 1 : column_Train-1);
T_train = signalTrain( : , column_Train);
% 提取测试集测试数据和目标类别
P_test = signalValidation( : , 1 : column_Validation-1);
T_test = signalValidation( : , column_Validation);

% svm分类
model = libsvmtrain(T_train,P_train,'-c 1 -g 0.0005');
% SVM网络预测
[predict_label, Accuracy, dec_values] = libsvmpredict(T_test, P_test, model);
%% 绘制混淆矩阵
cm = confusionmat(T_test', predict_label');
figure;
chart = heatmap(cm);
% 标签
label = {'安全环境','人员行走','轮式车辆','履带车辆'};
% title('Confusion Matrix');
% xlabel('Predicted Labels');
% ylabel('Actual Labels');
% 设置行和列的标签
chart.YDisplayLabels = label;
chart.XDisplayLabels = label;

% chart.FontName = 'Times New Roman';
colorbar;

%% 计算准确率、精度、召回率、F1 score

% 计算混淆矩阵
cunfusionMatrix = confusionmat(T_test, predict_label);

% 计算准确性、精确率、召回率
accuracy = sum(diag(cunfusionMatrix)) / sum(cunfusionMatrix(:)); % 准确性
precision = diag(cunfusionMatrix) ./ sum(cunfusionMatrix, 1)'; % 精确率 每一列的数据计算一个精确率
recall = diag(cunfusionMatrix) ./ sum(cunfusionMatrix, 2); % 召回率 每一行的数据计算一个召回率

% 计算 F1 曲线
F1_score = 2 * (precision .* recall) ./ (precision + recall);