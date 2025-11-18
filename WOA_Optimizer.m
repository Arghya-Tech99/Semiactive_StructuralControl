%% WOA_Optimizer.m: Whale Optimization Algorithm for COLQR Tuning

clc; % Clear command window

disp('--- Starting Whale Optimization Algorithm for COLQR Tuning ---');

%%  1. Run NSAE and Dynamic Model Setup
% Ensures system matrices (A, Bc, E) and seismic input (ag, t) are available
% generate_nsae_kanai_tajimi_2023Brandao;
% Dynamic_model; 

%% 2. Define Problem Dimensions and Constraints
n = size(Ms, 1);             % Number of stories (10)
N_STATES = 2 * n;            % 20 states
N_CONTROLS = size(Gamma, 2); % 1 control input
F_MR_MAX = 2.0 * 10^4;       % Damper force constraint (50 kN)
F_MR_MIN = 0.0;              
% The time series input for Simulink
input_excite = timeseries(ag, t); 

%% 3. WOA PARAMETERS 
WOA_Params.PopulationSize = 10;   % Number of search agents (whales) 
WOA_Params.MaxIterations = 30;    % Maximum number of optimization cycles 
WOA_Params.Dim = 2;               % Dimension of the search space (log10(Q_param), log10(R_param))

%% 4. DEFINING SEARCH BOUNDS
% Bounds: [log10(Q_param)_min, log10(R_param)_min] to [log10(Q_param)_max, log10(R_param)_max]
WOA_Params.LowerBound = [4, -4]; % Q_param >= 1e4, R_param >= 1e-4
WOA_Params.UpperBound = [10, 2]; % Q_param <= 1e10, R_param <= 1e2

disp(['WOA initialized: PopSize=', num2str(WOA_Params.PopulationSize), ...
      ', Iterations=', num2str(WOA_Params.MaxIterations)]);

%% 5. LAUNCH WOA
% COLQR_CostFunction is passed as a function handle.
[BestPos_log, BestCost, ~] = WhaleOptimizationAlgorithm(@COLQR_CostFunction, WOA_Params);

%% 6. POST-PROCESSING RESULTS & FINAL LQR GAIN
Q_param_WOA = 10^BestPos_log(1); % Convert back from log
R_param_WOA = 10^BestPos_log(2);

% Recalculate the final optimal K_lqr using the best found parameters
Q_WOA = Q_param_WOA * eye(N_STATES);
R_WOA = R_param_WOA * eye(N_CONTROLS);
K_WOA = lqr(A, Bc, Q_WOA, R_WOA); % Gain Matrix

%% 7. DISPLAY FINAL RESULTS AND GAIN MATRIX
disp('----------------------------------------------------');
disp('WOA Optimization Final Results:');
disp(['Optimal Q Parameter (log10): ', num2str(BestPos_log(1))]);
disp(['Optimal R Parameter (log10): ', num2str(BestPos_log(2))]);
disp(['Optimal Q_param: ', num2str(Q_param_WOA, '%.2e')]);
disp(['Optimal R_param: ', num2str(R_param_WOA, '%.2e')]);
disp(['Minimum Cost (Max Drift): ', num2str(BestCost, '%.4f')]);

% Display the complete K_WOA matrix
fprintf('\n=== FINAL OPTIMAL LQR GAIN MATRIX (K_WOA) ===\n');
fprintf('Matrix Size: %dx%d\n\n', size(K_WOA));

% Display the complete matrix
disp('Complete K_WOA Matrix:');
disp(K_WOA);

% Display matrix statistics
fprintf('\nMatrix Statistics:\n');
fprintf('Minimum value:  %.6e\n', min(K_WOA(:)));
fprintf('Maximum value:  %.6e\n', max(K_WOA(:)));
fprintf('Mean value:     %.6e\n', mean(K_WOA(:)));
fprintf('Standard deviation: %.6e\n', std(K_WOA(:)));
fprintf('Frobenius norm: %.6e\n', norm(K_WOA, 'fro'));

% Display first few elements for quick inspection
fprintf('\nFirst 5x5 block of K_WOA:\n');
disp(K_WOA(1:min(5,size(K_WOA,1)), 1:min(5,size(K_WOA,2))));

fprintf('\nOptimal K_WOA, F_MR_MAX/MIN, and input_excite ready for Simulink.\n');
disp('----------------------------------------------------');

%% 8. EXPORT TO WORKSPACE FOR SIMULINK
% Export variables to base workspace for Simulink access
assignin('base', 'K_WOA', K_WOA);
assignin('base', 'Q_param_WOA', Q_param_WOA);
assignin('base', 'R_param_WOA', R_param_WOA);
assignin('base', 'F_MR_MAX_WOA', F_MR_MAX);
assignin('base', 'F_MR_MIN_WOA', F_MR_MIN);
assignin('base', 'input_excite_WOA', input_excite);

fprintf('K_WOA and related parameters exported to base workspace.\n');
fprintf('Use "K_WOA" in your Simulink model for WOA-optimized control.\n');

% Clean up temporary variables
clear WOA_Params BestPos_log BestCost n N_STATES N_CONTROLS;