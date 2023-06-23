classdef CrossEntropyCost
    methods(Static)
        function sum_cost = fn(a, y)
            % Return the cost associated with an output ``a`` and desired output
            % ``y``.  Note that isnan is used to ensure numerical
            % stability.  In particular, if both ``a`` and ``y`` have a 1.0
            % in the same slot, then the expression (1-y)*log(1-a)
            % returns nan. The operation below ensures that that is converted
            % to the correct value (0.0).
            value = -y.*log(a)-(1-y).*log(1-a);
            value(isnan(value))=0;
            sum_cost = sum(value);
        end
        
        function value = delta(z, a, y)
            % Return the error delta from the output layer.  Note that the
            % parameter ``z`` is not used by the method.  It is included in
            % the method's parameters in order to make the interface
            % consistent with the delta method for other cost classes.
            value = a - y;
        end
    end
end