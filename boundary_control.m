function [X] = boundary_control(sizepop, dim, X, ub, lb)
    %边界控制
    for m = 1:sizepop
        for n = 1: dim
            if(X(m,n) > ub(n))
                X(m,n) = ub(n);
            end
            if(X(m,n) < lb(n))
                X(m,n) = lb(n);
            end
        end
    end
end

