classdef QuadraticCost
    methods(Static)
        function value = fn(A, Y)
            % Return the cost associated with an output ``a`` and desired 
            % output ``y``.
            value = (A-Y).^2;
        end
        
        function value = delta(z, A, Y)
            value = A - Y;
        end
    end
end