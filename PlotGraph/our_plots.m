TASK = 'MOTOR'
n_CONDITIONS = 5% number of conditions existing for that task (Matlab notation)
SUBJECT = 2  % in Matlab notation !!!!

for condition=1:n_CONDITIONS
    % Plotting Deep KO (if you want another, change it at PlotGraph)
    PlotGraph(TASK, condition, SUBJECT, false, true)
    
    %Plotting GLM controlled betas (if you want another, change it at PlotGraph)
    PlotGraph(TASK, condition, SUBJECT, false, false)
    
end