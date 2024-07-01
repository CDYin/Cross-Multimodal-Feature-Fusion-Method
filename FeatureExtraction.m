%% 提取震动信号的CWT谱图和MFCC谱图特征
clear;
clc;

% 加载数据
disp('开始加载数据！');
seismicDataFile = '.\data\sourceData\seismicData.xlsx';% 震动数据路径
audioDataList = dir('.\data\sourceData\audioData\*.wav');% 声数据文件夹路径
audioDataFiles=sort_nat({audioDataList.name});% 声文件名排序

seismicData = xlsread(seismicDataFile);% 加载震动数据
[row_seismic, column_seismic] = size(seismicData);

disp('数据加载完毕！');

%% 数据处理
feareturesfile = '.\data\featureData\fc-CBAM\fc-acoustic\features.xlsx';
trainfeaturefile = '.\data\featureData\fc-CBAM\fc-acoustic\train.xlsx';
testfeaturefile = '.\data\featureData\fc-CBAM\fc-acoustic\test.xlsx';
seismicpicfolder = '.\data\featureData\picdata\matlab\seismic\';
audiopicfolder = '.\data\featureData\picdata\matlab\audio\';

% 加载预训练网络
seismic_net = load('.\data\network\VGG\optimized\CBAM_VGG19_seismic_net.mat');
acoustic_net = load('.\data\network\VGG\optimized\CBAM_VGG19_acoustic_net.mat');
% lstm_seismic_net = load('.\data\network\LSTM\original\lstm_seismic_net.mat');
% lstm_acoustic_net = load('.\data\network\LSTM\original\lstm_acoustic_net.mat');
seismic_net = seismic_net.CBAM_VGG19_seismic_net;
acoustic_net = acoustic_net.CBAM_VGG19_acoustic_net;
% lstm_seismic_net = lstm_seismic_net.lstm_seismic_net;
% lstm_acoustic_net = lstm_acoustic_net.lstm_acoustic_net;
% 更新为特征提取网络中正确的层名称
featureLayer = 'fc7';
% featureLayer1 = 'fc1';
% featureLayer2 = 'fc_1';

disp('开始特征提取！');

for num = 1 : column_seismic
    % 震动数据和声数据分割
    seismicdata = seismicData( 1 : row_seismic-1 , num );
    [audiodata, fs] = audioread( strcat('.\data\sourceData\audioData\' , string(audioDataFiles{num})));
    % [audiodata_all, fs] = audioread( strcat('.\data\sourceData\audioData\' , string(audioDataFiles{num})));
    % 调整数据大小
    % audiodata = audiodata_all(1:16000,1);
    % 提取CWT特征图和MEL特征图
    pic_cwt = get_pic_CWT(seismicdata, 'amor');%morse, amor, bump
    pic_mel = get_pic_MEL(audiodata, fs);
    % 特征图保存到本地
    imwrite(pic_cwt,[seismicpicfolder,num2str(num),'.png'])
    imwrite(pic_mel, [audiopicfolder,num2str(num),'.png']);


    % 使用特征提取网络提取特征并构成特征向量
    seismic_features = activations(seismic_net, pic_cwt, featureLayer, 'OutputAs', 'rows');
    audio_features = activations(acoustic_net, pic_mel, featureLayer, 'OutputAs', 'rows');
    % lstm_seismic_features = activations(lstm_seismic_net, seismicdata, featureLayer1, 'OutputAs', 'rows');
    % lstm_audio_features = activations(lstm_acoustic_net, audiodata, featureLayer2, 'OutputAs', 'rows');
    % 特征写入Excel
    mix_feature = [seismic_features audio_features seismicData(row_seismic, num );];
    % mix_feature = [lstm_seismic_features lstm_audio_features seismicData(row_seismic, num );];

    if(num == 1)
        total_feature = mix_feature;
    else
        total_feature = [total_feature;mix_feature];
    end

    disp(['第' num2str(num) '个， 共' num2str(column_seismic) '个']);
end
xlswrite(feareturesfile, total_feature);
disp('特征提取完毕！');

%% 划分数据集并存入本地文件中
disp('开始划分数据！');

% 读取特征数据
all_features = xlsread( feareturesfile);
[features_row,features_column] = size(all_features);
% 划分训练集和测试集
randIndex = randperm(features_row);
% 划分训练集
train_num = floor(features_row*0.7);
test_num = features_row - train_num;
P_train = all_features(randIndex(1: train_num), 1:features_column-1);
T_train = all_features(randIndex(1: train_num), features_column);
% 划分测试集
P_test = all_features(randIndex(train_num+1: end), 1:features_column-1);
T_test = all_features(randIndex(train_num+1: end), features_column);
% 训练集和测试集写入本地文件
xlswrite(trainfeaturefile, [P_train T_train]);
xlswrite(testfeaturefile, [P_test T_test]);

disp('划分数据完毕！');