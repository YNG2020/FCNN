function [num_figure] = plot_monitor(num_figure, monitor_training_cost,...
    monitor_evaluation_cost,...
    evaluation_cost, training_cost)
    % Monitor the number of figure.
    
    epochs = max([length(evaluation_cost) length(training_cost)]);
    % Show the changes of the cost during training process.
    if monitor_training_cost && ~monitor_evaluation_cost
        num_figure = num_figure + 1;
        figure(num_figure)
        h1 = plot(1:epochs, training_cost, '-r','LineWidth',0.8);
        xlabel('Epoch', 'FontSize', 14)
        ylabel('Cost', 'FontSize', 14)
        legend(h1, 'Cost on training data','FontSize',10)
        legend('boxoff')

        num_figure = num_figure + 1;
        figure(num_figure)
        h1 = semilogy(1:epochs, training_cost, '-r','LineWidth',0.8);
        xlabel('Epoch', 'FontSize', 14)
        ylabel('Cost', 'FontSize', 14)
        legend(h1, 'Cost on training data','FontSize',10)
        legend('boxoff')
    elseif ~monitor_training_cost && monitor_evaluation_cost
        num_figure = num_figure + 1;
        figure(num_figure)
        h2 = plot(1:epochs, evaluation_cost,'-b','LineWidth',0.8);
        xlabel('Epoch', 'FontSize', 14)
        ylabel('Cost', 'FontSize', 14)
        legend(h2, 'Cost on test data','FontSize',10)
        legend('boxoff')

        num_figure = num_figure + 1;
        figure(num_figure)
        h1 = semilogy(1:epochs, evaluation_cost, '-b','LineWidth',0.8);
        xlabel('Epoch', 'FontSize', 14)
        ylabel('Cost', 'FontSize', 14)
        legend(h1, 'Cost on test data','FontSize',10)
        legend('boxoff')
    elseif monitor_training_cost && monitor_evaluation_cost
        num_figure = num_figure + 1;
        figure(num_figure)
        hold on
        h1 = plot(1:epochs, training_cost, '-r','LineWidth',0.8);
        h2 = plot(1:epochs, evaluation_cost,'-b','LineWidth',0.8);
        xlabel('Epoch', 'FontSize', 14)
        ylabel('Cost', 'FontSize', 14)
        legend([h1 h2], 'Cost on training data','Cost on test data','FontSize',10)
        legend('boxoff')

        num_figure = num_figure + 1;
        figure(num_figure)
        h1 = semilogy(1:epochs, training_cost, '-r','LineWidth',0.8);
        hold on
        h2 = semilogy(1:epochs, evaluation_cost,'-b','LineWidth',0.8);
        xlabel('Epoch', 'FontSize', 14)
        ylabel('Cost', 'FontSize', 14)
        legend([h1 h2], 'Cost on training data','Cost on test data','FontSize',10)
        legend('boxoff')
    end
end