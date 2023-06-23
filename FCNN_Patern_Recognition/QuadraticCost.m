classdef QuadraticCost
    methods(Static)
        function value = fn(a, y)
            % Return the cost associated with an output ``a`` and desired 
            % output ``y``.
            value = 0.5*norm(a-y).^2;
        end
        
        function value = delta(z, a, y)
            value = (a-y) .* sigmoid_prime(z);
        end
    end
end