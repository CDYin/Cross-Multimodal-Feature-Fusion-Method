function [acoustic_accuracy, lstm_acoustic_net, test_label, acoustic_predictedLabels] = Acoustic_LSTM_impl(Learning_rate,Hidden,Optimizer,BatchSize,Maxepoch)
% 加载数据
load('..\data\network\LSTM\acoustic_net_layer.mat')
seismicDataFile = '..\data\sourceData\seismicData.xlsx';% 震动数据路径
audioDataList = dir('..\data\sourceData\audioData\*.wav');% 声数据文件夹路径
audioDataFiles=sort_nat({audioDataList.name});% 声文件名排序

seismicData = xlsread(seismicDataFile);% 加载震动数据
[row_seismic, column_seismic] = size(seismicData);

% 数据处理
feareturesfile = '..\data\featureData\fc7\features.xlsx';
trainfeaturefile = '..\data\featureData\fc7\train.xlsx';
testfeaturefile = '..\data\featureData\fc7\test.xlsx';
seismicpicfolder = '..\data\featureData\picdata\seismic\';
audiopicfolder = '..\data\featureData\picdata\audio\';

% 原始序列
sequence = 1:column_seismic;
% 生成随机索引的排列
random_order = randperm(length(sequence));
% 根据随机索引重新排列序列
shuffled_sequence = sequence(random_order);

train_seismic = [];
train_acoustic = [];
test_seismic = [];
test_acoustic = [];
train_label = [];
test_label = [];

% 构建训练集、测试机、分类标签
for num = 1 : 1400
    % 震动数据和声数据分割
    data = seismicData( 1 : row_seismic-1 , shuffled_sequence(num) );
    [audiodata_all, fs] = audioread(strcat('..\data\sourceData\audioData\' , string(audioDataFiles{shuffled_sequence(num)})));
    % 调整数据大小
    audiodata = audiodata_all(1:16000,1);
    % 构建预训练数据集
    train_seismic = [train_seismic,data];
    train_acoustic = [train_acoustic,audiodata];
    train_label = [train_label,seismicData( row_seismic, shuffled_sequence(num) )];
end

for num = 1 : 600
    % 震动数据和声数据分割
    data = seismicData( 1 : row_seismic-1 , shuffled_sequence(1400+num) );
    [audiodata_all, fs] = audioread(strcat('..\data\sourceData\audioData\' , string(audioDataFiles{shuffled_sequence(1400+num)})));
    % 调整声数据大小
    audiodata = audiodata_all(1:16000,1);
    % 构建预训练数据集
    test_seismic = [test_seismic,data];
    test_acoustic = [test_acoustic,audiodata];
    test_label = [test_label,seismicData( row_seismic, shuffled_sequence(1400+num) )];
end

% 重塑训练集和测试机以满足网络输入条件
train_acoustic = double(reshape(train_acoustic, 16000,1,1,1400));
test_acoustic = double(reshape(test_acoustic, 16000,1,1,600));

% 数据格式转换
for  index = 1:1400
    p_train_acoustic{index,1} = train_acoustic(:,:,1,index);
end

for  index = 1:600
    p_test_acoustic{index,1} = test_acoustic(:,:,1,index);
end

% 训练标签转化为神经网络输入格式
train_label = categorical(train_label)';
test_label = categorical(test_label)';

% 用于LSTM的隐藏层个数
if Hidden==1
    HiddenNumber=32;
elseif Hidden==2
    HiddenNumber=64;
elseif Hidden==3
    HiddenNumber=128;
elseif Hidden==4
    HiddenNumber=256;
elseif Hidden==5
    HiddenNumber=512;
end

% 用于训练神经网络的求解器
if Optimizer==1
    SolverName='adam';         
elseif Optimizer==2
    SolverName='sgdm';         
elseif Optimizer==3
    SolverName='rmsprop';      
end

% 用于设置批大小
if BatchSize==1
    Batch=8;
elseif BatchSize==2
    Batch=16;
elseif BatchSize==3
    Batch=32;
elseif BatchSize==4
    Batch=64;
elseif BatchSize==5
    Batch=128;
end

% 用于训练神经网络的求解器
if Maxepoch==1
    epoch=10;    
    LearnRateDropPeriod = 7;
elseif Maxepoch==2
    epoch=20;     
    LearnRateDropPeriod = 14;
elseif Maxepoch==3
    epoch=30;      
    LearnRateDropPeriod = 21;
elseif Maxepoch==4
    epoch=40;     
    LearnRateDropPeriod = 28;
elseif Maxepoch==5
    epoch=50;   
    LearnRateDropPeriod = 35;
elseif Maxepoch==6
    epoch=60;   
    LearnRateDropPeriod = 42;
elseif Maxepoch==7
    epoch=70;  
    LearnRateDropPeriod = 49;
elseif Maxepoch==8
    epoch=80;         
    LearnRateDropPeriod = 56;
elseif Maxepoch==9
    epoch=90;   
    LearnRateDropPeriod = 63;
elseif Maxepoch==10
    epoch=100; 
    LearnRateDropPeriod = 70;
end

layers = [
    sequenceInputLayer(500,"Name","input")
    lstmLayer(HiddenNumber,"Name","lstm","OutputMode","last")
    dropoutLayer(0.5,"Name","dropout_1")
    fullyConnectedLayer(500,"Name","fc1")
    dropoutLayer(0.5,"Name","dropout_2")
    fullyConnectedLayer(4,"Name","fc2")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classification")];

% 设定训练参数
options = trainingOptions(SolverName, ...
    'MiniBatchSize',Batch, ...                       % 下一个函数返回的小批量的大小，指定为正整数。默认值为128。
    'Shuffle','every-epoch',...                      % 数据重排选项。在每个训练时期之前混洗训练数据，并且在每个神经网络验证之前混洗验证数据。
    'MaxEpochs',epoch, ...                              % 最大迭代轮次
    'InitialLearnRate',Learning_rate,...             % 初始全局学习率
    'LearnRateSchedule','piecewise',...              % 培训期间降低学习率的选项，LearnRateDropFactor和LearnRateDropPeriod设置。
    'LearnRateDropFactor', 0.02, ...                 % 指定降低初始学习率的因子
    'LearnRateDropPeriod', LearnRateDropPeriod, ...                    % 指定降低初始学习率轮数的因子
    'ValidationPatience', Inf, ...                   % 关闭验证
    'Verbose',false,...                              % 软件生成具有详细消息的代码
    'ExecutionEnvironment','gpu');                   % 用于训练神经网络的硬件资源

%训练网络
lstm_acoustic_net = trainNetwork(p_train_acoustic, train_label, acoustic_net_layer, options);
% 验证训练好的模型
acoustic_predictedLabels = classify(lstm_acoustic_net, p_test_acoustic);
acoustic_accuracy = sum(acoustic_predictedLabels ~= test_label) / numel(test_label);
end

