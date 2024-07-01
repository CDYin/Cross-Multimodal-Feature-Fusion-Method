%% 设置适应度函数并配置初始参数
% This function containts full information and implementations of the benchmark
% lb is the lower bound: lb=[lb_1,lb_2,...,lb_d]
% up is the uppper bound: ub=[ub_1,ub_2,...,ub_d]

function [lb,ub,dim,fobj] = Get_SSAFunction_details(F)
switch F
    case 'F1'
        fobj = @F1;
        %配置初始参数上下界值
        lb=[1e-4, 0.50, 0.50, 0.50, 0.50];
        ub=[1e-2, 5.49, 3.49, 5.49, 10.49];
        % dim是变量的数量(问题的维数)
        dim=5;
end
end

% F1 适应度函数实现
function Accuracy = F1(x)
%四舍五入为整数
x(2)=int8(x(2));   % 约束隐藏数为整数
x(3)=int8(x(3));   % 约束优化算法符合if语句
x(4)=int8(x(4));   % 约束批大小为整数
x(5)=int8(x(5));   % 约束最大训练次数符合if语句

% disp(x(:));
Accuracy = Seismic_LSTM_impl(x(1),x(2),x(3),x(4),x(5));
% Accuracy = Acoustic_LSTM_impl(x(1),x(2),x(3),x(4),x(5));
end

