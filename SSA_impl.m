%% 麻雀搜索算法实现
function [Best_pos,Best_score,curve]=SSA_impl(sizepop,iter_max,lb,ub,dim,fobj)

%-----------------初始化--------------------------%
ST = 0.75;            %预警值
PD = 0.7;             %发现者的比列，剩下的是加入者
SD = 0.4;             %意识到有危险麻雀的比重

PDNumber = sizepop*PD;    %发现者数量
SDNumber = sizepop*SD;    %意识到有危险麻雀数量

% 种群初始化
X=SSA_initialization(sizepop,dim,ub,lb);
% 计算初始适应度值
fitness = zeros(1,sizepop);
disp('计算初始适应度值');
for i = 1:sizepop
    fitness(i) =  fobj(X(i,:));
    disp(['共' num2str(sizepop) '个, 第' num2str(i) '个初始适应度值= ' num2str(fitness(i))]);
end
disp('初始适应度值计算完毕');

pFit = fitness;
[ fMin, bestI ] = min( fitness );      % fMin表示全局最优适应度值
bestX = X( bestI, : );                 % bestX表示与fMin相对应的全局最佳位置

% 麻雀搜索算法开始迭代优化
disp('麻雀搜索算法开始迭代优化');
for iter = 1: iter_max
    
    [ ~, sortIndex ] = sort( pFit );% Sort.
    
    [fmax,worseI]=max( pFit );
    worse= X(worseI,:);
    
    %安全值
    R2 = rand(1);
    %发现者位置更新
    if(R2<ST)
        for i = 1 : PDNumber                                                   % Equation (3)
            r1=rand(1);
            X( sortIndex( i ), : ) = X( sortIndex( i ), : )*exp(-(i)/(r1*iter_max));
            X =  boundary_control(sizepop, dim, X, ub, lb);
            fitness(sortIndex( i ))=fobj(X(sortIndex( i ),:));
        end
    else
        for i = 1 : PDNumber
            X( sortIndex( i ), : ) = X( sortIndex( i ), : ) + randn(1)*ones(1,dim);
            X =  boundary_control(sizepop, dim, X, ub, lb);
            fitness(sortIndex( i ))=fobj( X(sortIndex( i ),:) );
        end
    end
    % 更新最优位置
    [ ~, bestII ] = min( fitness );
    bestXX = X( bestII, : );
    %跟随者位置更新
    for i = ( PDNumber + 1 ) : sizepop                     % Equation (4)
        
        A=floor(rand(1,dim)*2)*2-1;
        
        if( i>(sizepop/2))
            X( sortIndex(i ), : )=randn(1)*exp((worse-X( sortIndex( i ), : ))/(i)^2);
        else
            X( sortIndex( i ), : )=bestXX+(abs(( X( sortIndex( i ), : )-bestXX)))*(A'*(A*A')^(-1))*ones(1,dim);
        end
        X =  boundary_control(sizepop, dim, X, ub, lb);
        fitness(sortIndex( i ))=fobj(X(sortIndex( i ),:));
    end
    
    randNum=randperm(numel(sortIndex));
    SDpop=sortIndex(randNum(1:SDNumber));
    %危险更新
    for j = 1 : SDNumber      % Equation (5)
        if( pFit( sortIndex( SDpop(j) ) )>(fMin) )
            X( sortIndex( SDpop(j) ), : )=bestX+(randn(1,dim)).*(abs(( X( sortIndex( SDpop(j) ), : ) -bestX)));
        else
            X( sortIndex( SDpop(j) ), : ) =X( sortIndex( SDpop(j) ), : )+(2*rand(1)-1)*(abs(X( sortIndex( SDpop(j) ), : )-worse))/ ( pFit( sortIndex( SDpop(j) ) )-fmax+1e-50);
        end
        X =  boundary_control(sizepop, dim, X, ub, lb);
        fitness(sortIndex( i ))=fobj(X(sortIndex( i ),:));
    end

    X =  boundary_control(sizepop, dim, X, ub, lb);
    
    %更新位置
    for i = 1 : sizepop
        if ( fitness( i ) < pFit( i ) )
            pFit( i ) = fitness( i );
            X(i,:) = X(i,:);
        end
        
        if( pFit( i ) < fMin )
            fMin = pFit( i );
            bestX = X( i, : );
        end
    end
    curve(iter) = fMin;
    
    disp(['第' num2str(iter) '个， 共' num2str(iter_max) '个']);
    disp(['最优值=' num2str(fMin)]);
    disp(['最优解=' num2str(bestX)]);
end
disp('麻雀搜索算法开始迭代优化完成');
Best_pos =bestX;
Best_score = curve(end);
disp(['最优值=' num2str(Best_score)]);
disp(['最优解=' num2str(Best_pos)]);
end