% Simulate visual evoked respones using SEREEGA toolbox
% peak times and width based on literature (Luck, 2005; ISBN 0-262-62196-7)

%% SET THESE TO YOUR LOCAL PROJECT PATHS

PROJECT_PATH = "C:\Users\micha\projects\oscillation_vs_exponent";
SEREEGA_PATH = "C:\Users\micha\matlab/SEREEGA-master";

%% Set parameters

% signal parameters
FS = 2000; % sampling frequnecy (Hz)
TIME_OFFSET = 200; % time offset (ms). if epoch start time != 0
EPOCH_LENGTH = 1000; % ms
N_TRIALS = 100; % number of trials to simulate

% visual evoked response parameters  
% P1
p1_peak_time = [100, 130]; % [lower_bound, upper_bound]
p1_peak_width = [80, 4]; % [mean, std]
p1_peak_amp = [100, 5]; %[mean, std]

% N1
n1_peak_time = [150, 200]; % [lower_bound, upper_bound]
n1_peak_width = [100, 5]; % [mean, std]
n1_peak_amp = [-100, 5]; % [mean, std]

%% simulate evoked resopnse and save to file

% Add SEREEGA to path
addpath(genpath(SEREEGA_PATH))

% simulate
evoked = [];
for ii = 1:N_TRIALS
    % generate peak parameters 
    peak_times = [random('Uniform', p1_peak_time(1), p1_peak_time(2)),...
                  random('Uniform', n1_peak_time(1), n1_peak_time(2))];
    peak_widths = [random('Normal', p1_peak_width(1), p1_peak_width(2)),...
                   random('Normal', n1_peak_width(1), n1_peak_width(2))];
    peak_amplitudes = [random('Normal', p1_peak_amp(1), p1_peak_amp(2)) ...
                       random('Normal', n1_peak_amp(1), n1_peak_amp(2))];

    % simulate evoked response with SEREEGA
    evoked(ii,:) = erp_generate_signal(peak_times+TIME_OFFSET, peak_widths, ...
                                       peak_amplitudes, FS, EPOCH_LENGTH);
end

% save results
save(fullfile(PROJECT_PATH, 'data\simulated_evoked_response', ...
    'visual_evoked_response'), 'evoked')



