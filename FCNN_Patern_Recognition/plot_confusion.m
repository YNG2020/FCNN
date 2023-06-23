function [] = plot_confusion(net, ...
    input, output, ...
    test_input, test_output, ...
    valid_input, valid_output)

    if (nargin < 6)
        valid_input = 0;
        valid_output = 0;
    end
    
    [pro_tr_input,PS] = mapminmax(input);
    [pro_te_input] = mapminmax('apply', test_input, PS);
    [pro_va_input] = mapminmax('apply', valid_input, PS);

    tr_predict = net_ouput(net, pro_tr_input, output);
    te_predict = net_ouput(net, pro_te_input, test_output);
    
    if (nargin >= 6)
        va_predict = net_ouput(net, pro_va_input, valid_output);
    else
        valid_output = [];
        va_predict = [];
    end
    
    all_output = [output test_output valid_output];
    all_predict = [tr_predict te_predict va_predict];
    
    if (nargin >= 6)
        plotconfusion(output,tr_predict,'Training', ...
            test_output,te_predict, 'Test', ...
            valid_output,va_predict, 'Validation', ...
            all_output, all_predict, 'All');
    else
        plotconfusion(output,tr_predict,'Training', ...
            test_output,te_predict, 'Test', ...
            all_output, all_predict, 'All');
    end
    
end