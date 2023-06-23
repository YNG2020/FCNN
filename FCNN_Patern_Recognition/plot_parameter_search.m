function [num_figure] = plot_parameter_search(num_figure, accuracy, eta_best, eta_space)
    
    num_figure = num_figure + 1;
    figure(num_figure)
    h1 = plot(eta_space, accuracy, '-r','LineWidth',0.8);
    hold on
    xlabel('Eta', 'FontSize', 14)
    ylabel('Cost', 'FontSize', 14)
    title({strcat('Best eta = ',num2str(eta_best))},'FontSize',10)
    legend(h1, 'Cost on the data','FontSize',10)
    legend('boxoff')
end