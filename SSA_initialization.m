%% 此函数初始化搜索代理的第一次填充
function Positions=SSA_initialization(SearchAgents_num,dim,ub,lb)
Positions = zeros(SearchAgents_num, dim);
% Positions(1,:)=rand(1,dim);
% % 使用circle混沌映射初始化种群
% for i=2:SearchAgents_num
%     Positions(i,:)=mod(Positions(i-1,:)+0.2-(0.25./pi).*sin(2.*pi.*Positions(i-1,:)),1);%Circle混沌映射
% end
% % 将种群变量大小约束到最大最小值之间
% for i=1:dim
%     ub_i=ub(i);
%     lb_i=lb(i);
%     for j = 1:SearchAgents_num
%         % 每个变量的范围在最大最小值之间
%         if (Positions(j,i)<lb_i) || (Positions(j,i)>ub_i)
%             Positions(j,i)=rand().*(ub_i-lb_i)+lb_i;
%         end
%     end
% end

% 将种群变量大小约束到最大最小值之间
for i=1:dim
    ub_i=ub(i);
    lb_i=lb(i);
    % 每个变量的范围在最大最小值之间
    Positions(:,i)=rand(SearchAgents_num,1).*(ub_i-lb_i)+lb_i;
end