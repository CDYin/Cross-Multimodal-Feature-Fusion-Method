%% SSA优化LSTM

% fobj                  设定适应度函数
% dim                   设定维度
% Max_iteration         设定最大迭代次数
% SearchAgents_num      种群数量
% lb                    变量下边界
% ub                    变量上边界

% 主函数开始运行
SearchAgents_num=20;  % 种群数量
Function_name='F1';   % 设定适应度函数
Max_iteration=50;     % 设定最大迭代次数
% 设定边界以及优化函数
disp('开始设定边界以及优化函数');
[lb, ub, dim, fobj] = Get_SSAFunction_details(Function_name);  
disp('边界以及优化函数设定完毕');
% 开始优化参数
disp('开始优化参数');
[Best_pos, Best_score, SSA_curve] = SSA_impl(SearchAgents_num, Max_iteration, lb, ub, dim, fobj); 
disp('参数优化完毕');
% 得到最优的训练网络结构
disp('开始训练最优LSTM网络');
Best_pos(2:dim) = int8(Best_pos(2:dim));
[Accuracy, net, T_test, predictedLabels] = Seismic_LSTM_impl(Best_pos(1), Best_pos(2), Best_pos(3), Best_pos(4), Best_pos(5));
% [Accuracy, net, T_test, predictedLabels] = Acoustic_LSTM_impl(Best_pos(1), Best_pos(2), Best_pos(3), Best_pos(4), Best_pos(5));
disp('LSTM网络训练训练完毕');

%% 麻雀搜索算法优化曲线
figure;
plot(SSA_curve,'Color','r')
axis([1 Max_iteration 0.32 0.36])
title('Seismic Fitness curve')
xlabel('Iteration');
ylabel('Best Fitness obtained so far');

grid on
box on
legend('SSA')

display(['The best solution obtained by SSA is : ', num2str(Best_pos)]);
display(['The best optimal value of the objective funciton found by SSA is : ', num2str(Best_score)]);

%% 绘制混淆矩阵和准确率点图
figure
cm = confusionchart(T_test, predictedLabels);
cm.Title = 'Confusion Matrix for Test Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';

figure
Num_Test = size(T_test,1);
plot(1: Num_Test, T_test, 'r-*', 1: Num_Test, predictedLabels, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'测试集预测结果对比'; ['准确率=' num2str((1-Accuracy)*100) '%']};
title(string)
xlim([1 Num_Test])
grid


