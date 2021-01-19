%% This script computes rsfMRI instantaneous phase synchrony and multilayer modularity switching during brain stimulation
%  v1 - January 2021: Mangor Pedersen - Auckland University of Technology (AUT), New Zealand.

%% Dependencies
% note: 'y_Read' and 'y_ExtractROISignal' can be found at http://rfmri.org/dpabi
% note: 'postprocess_ordinal_multilayer' 'multiord' and 'iterated_genlouvain' can be founds at https://github.com/GenLouvain/GenLouvain

%% Preliminaries and data-path
fMRI_data_path = dir(['*','postop','*']); % directory with all rsfMRI stimulation data %%% find all the post-op data containing brain stimulation epochs
fMRI_data_path(14) = []; % scan too short for analysis (less than 16 epochs)
fMRI_data_path(52-1) = []; % scan too short for analysis (less than 16 epochs)
fMRI_data_path(150-2) = []; % scan too short for analysis (less than 16 epochs)

N_subs = length(fMRI_data_path); % number of subjects for analysis
omga = [0.1 1 2 3]; % omega parameters
gmma = [1 1.1 1.2 1.3]; % gamma parameters
network_density_thr = [0.1 0.15 0.2]; % network density parameters
n_rand = 100; % number of permutations - for temporal null model
NoStim_epoch = 1:2:15; % NoStim epoch numbers
Stim_epoch = 2:2:16; % Stim epoch numbers

%% Pre-allocate cells and arrays
NoStim = zeros(N_subs,Stim_epoch(end),length(network_density_thr),length(gmma),length(omga),196);
Stim = zeros(N_subs,Stim_epoch(end),length(network_density_thr),length(gmma),length(omga),196);
NoStim_rand = zeros(N_subs,Stim_epoch(end),n_rand,length(network_density_thr),length(gmma),length(omga),196);
Stim_rand = zeros(N_subs,Stim_epoch(end),n_rand,length(network_density_thr),length(gmma),length(omga),196);
switching = cell(length(network_density_thr),length(gmma),length(omga),N_subs);
switching_rand = cell(length(network_density_thr),length(gmma),length(omga),N_subs);
Q_value = zeros(length(network_density_thr),length(gmma),length(omga),N_subs);
n_modules = zeros(length(network_density_thr),length(gmma),length(omga),N_subs);
timings = cell(1,N_subs);

%% Calculate network switching between epochs with stimulation vs epochs with no stimulation
for n = 1:N_subs % loop over all participants
    % load filtered fMRI data
    [fMRI_data, Header] = y_Read(fMRI_data_path(n).name);
    ROISignals = y_ExtractROISignal(fMRI_data,{y_Read('path_to_parcellation_mask')},'output_name.nii',...
        y_Read('path_to_binary_mask'),1,1,[],[],[],[],[],Header,5);
    
    % load intracranial stimulaiton timing
    if exist((['path_to_stimulation_data' fMRI_data_path(n).name(1:7) '\ses-postop\func\' fMRI_data_path(n).name(1:33) '_events.tsv']),'file')
        temp_tbl = tdfread(['path_to_stimulation_data' fMRI_data_path(n).name(1:7) '\ses-postop\func\' fMRI_data_path(n).name(1:33) '_events.tsv']);
        timings{n} = round(temp_tbl.onset./3);
    else
        timings{n} = timings{n-1}; % if timing information is not available
    end
    
    total_nodes = size(ROISignals,2); % total number of nodes
    total_timepoints = size(ROISignals,1); % total number of time-points
    
    inst_phase = angle(hilbert(ROISignals)); % calculate instantaneous phase of fMRI data
    
    for thr = 1:length(network_density_thr) % loop over all density thresholds
        A = cell(1,total_timepoints); % pre-allocate cell to store instantaneous phase tensors for all time-points
        
        for nn = 1:total_timepoints % loop over all time-points
            W = 1-abs(sin(bsxfun(@minus,inst_phase(nn,:)',inst_phase(nn,:)))); % instantaneous phase synchrony, for each time-point
            W = W - eye(size(W,1)); % set matrix diagonal to 0
            W = weight_conversion(threshold_proportional(W,network_density_thr(thr)),'binarize'); % threshold/binarize matrices, for each time-point
            A{nn} = W; % store each matrix in a cell (this is the input to the multilayer modularity model)
            
        end % end time-points
        
        for p1 = 1:length(gmma) % loop over each pre-defined gamma threshold
            for p2 = 1:length(omga) % loop over each pre-defined omega threshold
                reverseStr = ''; theElapsedTime = tic; % estimate how long each multilayer modularity model takes
                
                [B,mm] = multiord(A,gmma(p1),omga(p2)); % set multilayer modularity model with pre-defined omega and gamma values (A = time-varying connectivity)
                PP = @(S) postprocess_ordinal_multilayer(S,total_timepoints); % use an ordinal multilayer network model
                [multilayer_network,Q1,~] = iterated_genlouvain(B,10000,0,1,'moverandw',[],PP); % use an iterative modularity appproach (based on the Louvain algorithm)
                Q_value(thr,p1,p2,n) = Q1/mm; % Q-value for each network
                multilayer_network = reshape(multilayer_network,total_nodes,total_timepoints); % transform output modules to a 2D array
                
                multilayer_network_rand = zeros(total_nodes,total_timepoints,n_rand); % pre-allocate temporal null model
                
                for iii = 1:n_rand 
                    A_rand = A(randperm(size(A,2))); % randomly shuffle time-points
                    % calculate multilayer modularity for each randomization (line #82-84 is same procedure as line #71-75)
                    [B_rand,~] = multiord(A_rand,gmma(p1),omga(p2));
                    PP = @(S) postprocess_ordinal_multilayer(S,total_timepoints);
                    multilayer_network_rand(:,:,iii) = reshape(iterated_genlouvain(B_rand,10000,0,1,'moverandw',[],PP),total_nodes,total_timepoints); 
    
                end % end randomizations
                
                % pre-allocate arrays
                switching_temp = zeros(total_nodes,total_timepoints);
                switching_rand_temp = zeros(total_nodes,total_timepoints,n_rand);
                
                for node_index = 1:total_nodes % loop over all nodes
                    for t = 1:total_timepoints-1 % loop over all time-points
                        % find nodal network switching between between adjacent time-points
                        t1_and_t2 = multilayer_network(:,t:t+1)'; % vectorise the adjacent time point t and t+1
                        adjacent_switch = t1_and_t2(2,:) - t1_and_t2(1,:); % detect any network switching between two adjacent time-points
                        adjacent_switch(adjacent_switch~=0) = 1; % binarize modular switching between time-points
                        switching_temp(node_index,t) = adjacent_switch(node_index); % store the final switching information for node_ind, at time-point, t
                        
                        for tt = 1:n_rand % loop over all randomizations
                            multilayer_network_rand_loop = squeeze(multilayer_network_rand(:,:,tt)); % modularity decomposition for each randomization
                            % find nodal network switching between between adjacent time-points, for each randomization (line #103-106 is the same procedure as line #95-98)
                            t1_and_t2_rand = multilayer_network_rand_loop(:,t:t+1)';
                            adjacent_switch_rand = t1_and_t2_rand(2,:) - t1_and_t2_rand(1,:);
                            adjacent_switch_rand(adjacent_switch_rand~=0) = 1;
                            switching_rand_temp(node_index,t,tt) = adjacent_switch_rand(node_index);
                            
                        end % end randomizations
                    end % end time-points
                end % end nodes
                
                % store all nodal network switching between between adjacent time-points, for original and random networks
                switching{thr,p1,p2,n} = switching_temp;
                switching_rand{thr,p1,p2,n} = switching_rand_temp;
                
                % on-screen information
                msg = sprintf('\n Q %g; sub %d/%d; dens %g; gamma %g; omega %g; t %d; elapsed t %g min\n',...
                    Q1/mm,n,N_subs,network_density_thr(thr),gmma(p1),omga(p2),total_timepoints,toc(theElapsedTime)/60); fprintf([reverseStr,msg]);
                
            end % end omega parameters
        end % end gamma parameters
    end % end network densities
end % end subject

%% Calculate average network switching within each epoch
for thr = 1:length(network_density_thr) % loop over each pre-defined network density threshold
    for p1 = 1:length(gmma) % loop over each pre-defined gamma threshold
        for p2 = 1:length(omga) % loop over each pre-defined omega threshold
            for i = 1:N_subs % loop over each participant
                timings_temp = cell2mat(timings(i)); % Find stimulation timings for participant # i
                
                % find accurate timing information, for each of the 16 epochs
                epoch_timing{1} = 1:10;
                epoch_timing{2} = timings_temp(1):timings_temp(10);
                epoch_timing{3} = timings_temp(10)+2:(timings_temp(10)+2)+9;
                epoch_timing{4} = timings_temp(11):timings_temp(20);
                epoch_timing{5} = timings_temp(20)+2:(timings_temp(20)+2)+9;
                epoch_timing{6} = timings_temp(21):timings_temp(30);
                epoch_timing{7} = timings_temp(30)+2:(timings_temp(30)+2)+9;
                epoch_timing{8} = timings_temp(31):timings_temp(40);
                epoch_timing{9} = timings_temp(40)+2:(timings_temp(40)+2)+9;
                epoch_timing{10} = timings_temp(41):timings_temp(50);
                epoch_timing{11} = timings_temp(50)+2:(timings_temp(50)+2)+9;
                epoch_timing{12} = timings_temp(51):timings_temp(60);
                epoch_timing{13} = timings_temp(60)+2:(timings_temp(60)+2)+9;
                epoch_timing{14} = timings_temp(61):timings_temp(70);
                epoch_timing{15} = timings_temp(70)+2:(timings_temp(70)+2)+9;
                epoch_timing{16} = timings_temp(71):timings_temp(80);
                
                % retrieve switching information, for subject i, and the specific network parameters within the loop
                network_switching_array = cell2mat(switching(thr,p1,p2,i));
                network_switching_array_rand = cell2mat(switching_rand(thr,p1,p2,i));
                
                % Percentage network switches within each epoch (Stim and NoStim) divided by total possible network switches
                for ii = 1:length(NoStim_epoch) % loop over all epochs
                    NoStim(i,ii,thr,p1,p2,:) = nansum(network_switching_array(:,cell2mat(epoch_timing(NoStim_epoch(ii)))),2) ... % number of 'switches' within each epoch (NoStim)
                        ./ size(network_switching_array(:,cell2mat(epoch_timing(NoStim_epoch(ii)))),2).*100; % divided by total possible network switches
                    
                    Stim(i,ii,thr,p1,p2,:) = nansum(network_switching_array(:,cell2mat(epoch_timing(Stim_epoch(ii)))),2) ... % number of 'switches' within each epoch (NoStim)
                        ./ size(network_switching_array(:,cell2mat(epoch_timing(Stim_epoch(ii)))),2).*100; % divided by total possible network switches
                    
                    % Percentage network switches within each epoch (Stim and NoStim) divided by total possible network switches, for each randomisation
                    for iii = 1:n_rand
                        NoStim_rand(i,ii,iii,thr,p1,p2,:) = nansum(network_switching_array_rand(:,cell2mat(epoch_timing(NoStim_epoch(ii))),iii),2) ... % number of 'switches' within each epoch (NoStim)
                            ./ size(network_switching_array_rand(:,cell2mat(epoch_timing(NoStim_epoch(ii)))),2).*100; % divided by total possible network switches
                        
                        Stim_rand(i,ii,iii,thr,p1,p2,:) = nansum(network_switching_array_rand(:,cell2mat(epoch_timing(Stim_epoch(ii))),iii),2) ... % number of 'switches' within each epoch (Stim)
                            ./ size(network_switching_array_rand(:,cell2mat(epoch_timing(Stim_epoch(ii)))),2).*100; % divided by total possible network switches
                        
                    end % end randomizations
                end % end epochs
            end % end subject
        end % end omega parameters
    end % end gamma parameters
end % end network densities

%% Display swithing data for each epoch (original = top; randomized = bottom)
% calculate the mean swithing rate within each epoch across all randomizations
Stim_rand_mean = squeeze(nanmean(Stim_rand,3));
NoStim_rand_mean = squeeze(nanmean(NoStim_rand,3));

% plot swithcing scores for each epoch ('notBoxPlot' is found at https://au.mathworks.com/matlabcentral/fileexchange/26508-notboxplot)
figure; subplot(2,1,1); hold on
notBoxPlot([squeeze(nanmean(NoStim(:,1,:),3)) squeeze(nanmean(Stim(:,1,:),3)) ...
    squeeze(nanmean(NoStim(:,2,:),3)) squeeze(nanmean(Stim(:,2,:),3)) ...
    squeeze(nanmean(NoStim(:,3,:),3)) squeeze(nanmean(Stim(:,3,:),3)) ...
    squeeze(nanmean(NoStim(:,4,:),3)) squeeze(nanmean(Stim(:,4,:),3)) ...
    squeeze(nanmean(NoStim(:,5,:),3)) squeeze(nanmean(Stim(:,5,:),3)) ...
    squeeze(nanmean(NoStim(:,6,:),3)) squeeze(nanmean(Stim(:,6,:),3)) ...
    squeeze(nanmean(NoStim(:,7,:),3)) squeeze(nanmean(Stim(:,7,:),3)) ...
    squeeze(nanmean(NoStim(:,8,:),3)) squeeze(nanmean(Stim(:,8,:),3)) ...
    squeeze(nanmean(nanmean(NoStim,3),2)) squeeze(nanmean(nanmean(Stim,3),2))]);  hold off

subplot(2,1,2); hold on
notBoxPlot([squeeze(nanmean(NoStim_rand_mean(:,1,:),3)) squeeze(nanmean(Stim_rand_mean(:,1,:),3)) ...
    squeeze(nanmean(NoStim_rand_mean(:,2,:),3)) squeeze(nanmean(Stim_rand_mean(:,2,:),3)) ...
    squeeze(nanmean(NoStim_rand_mean(:,3,:),3)) squeeze(nanmean(Stim_rand_mean(:,3,:),3)) ...
    squeeze(nanmean(NoStim_rand_mean(:,4,:),3)) squeeze(nanmean(Stim_rand_mean(:,4,:),3)) ...
    squeeze(nanmean(NoStim_rand_mean(:,5,:),3)) squeeze(nanmean(Stim_rand_mean(:,5,:),3)) ...
    squeeze(nanmean(NoStim_rand_mean(:,6,:),3)) squeeze(nanmean(Stim_rand_mean(:,6,:),3)) ...
    squeeze(nanmean(NoStim_rand_mean(:,7,:),3)) squeeze(nanmean(Stim_rand_mean(:,7,:),3)) ...
    squeeze(nanmean(NoStim_rand_mean(:,8,:),3)) squeeze(nanmean(Stim_rand_mean(:,8,:),3)) ...
    squeeze(nanmean(nanmean(NoStim_rand_mean,3),2)) squeeze(nanmean(nanmean(Stim_rand_mean,3),2))]); hold off