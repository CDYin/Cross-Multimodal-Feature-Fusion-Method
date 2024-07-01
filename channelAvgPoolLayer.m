classdef channelAvgPoolLayer < nnet.layer.Layer & nnet.layer.Acceleratable% ...
    % & nnet.layer.Formattable ... % (Optional)
    % & nnet.layer.Acceleratable % (Optional)

    properties
        % (Optional) Layer properties.

        % Declare layer properties here.
    end

    properties (Learnable)
        % (Optional) Layer learnable parameters.

        % Declare learnable parameters here.
    end

    properties (State)
        % (Optional) Layer state parameters.

        % Declare state parameters here.
    end

    properties (Learnable, State)
        % (Optional) Nested dlnetwork objects with both learnable
        % parameters and state parameters.

        % Declare nested networks with learnable and state parameters here.
    end

    methods
        function this = channelAvgPoolLayer(name)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            this.Name = name;
            % Define layer constructor function here.
        end

        function X = predict(this,X)
            % 获取通道数和图像大小
            [H, W, ~,B] = size(X);

            % 遍历每个通道
            for b = 1:B
                for h = 1:H
                    for w = 1:W
                        % 将池化后的结果存入输出张量中
                        X(h, w, 1,b) = mean(X(h, w, :,b));
                    end
                end
            end
            X=X(1:H,1:W,1,1:B);
        end
    end
end