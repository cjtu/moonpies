function [totalIceS,ageVecs,ageMapS,stratMatrixEjectaS] = iceStratModel_rev2_south(xyCells,runNum,ejS,row,col)

load(['rngStates/state' num2str(runNum)]);
rng(rngstate);

f24 = 'false';

%% Fixed Parameters
pctAreaS = 1.30E+04;
hopEfficS = 0.054; % From modified Kloos model with Hayne area
% pctAreaN = 5.30E+03;
% hopEfficN = 0.027; % From modified Kloos model with Hayne area

xScale = -400:(800/(xyCells-1)):400; % grid is 800x800 km or ~75 degrees to pole

%% Set up
% Load data for craters
load(strcat(['craterData' 'S']));
craterLatS = craterData(:,1);
craterLonS = craterData(:,2);
craterDiamS = craterData(:,3); % [km]
craterAgeS = craterData(:,4); % [Ga]
craterAgeS = craterAgeInstance(craterAgeS,craterData(:,5),craterData(:,6));

% load(strcat(['craterData' 'N']));
% craterLatN = craterData(:,1);
% craterLonN = craterData(:,2);
% craterDiamN = craterData(:,3); % [km]
% craterAgeN = craterData(:,4); % [Ga]
% craterAgeN = craterAgeInstance(craterAgeN,craterData(:,5),craterData(:,6));

% Timesteps are 10 Myr
numTimesteps = 425;
timesteps = flipud((0:0.01:4.25)');

% Make empty maps
stratColVulxIceS = zeros(numTimesteps,1); % ice from volcanism for each timestep [m]
stratColImpactIceS = zeros(numTimesteps,1); % ice from impacts for each timestep [m]
totalIceS = zeros(numTimesteps,1); % total icer for each timestep with erosion [m]
stratMatrixEjectaS = zeros(xyCells,xyCells,numTimesteps); % ejecta thickness for each location and timestep [m]
totalEjectaMapS = zeros(xyCells,xyCells); % total ejecta thickness for each location
ageMapS = ones(xyCells,xyCells)*10; % age of youngest crater for each location [Ga]

% stratColVulxIceN = zeros(numTimesteps,1); % ice from volcanism for each timestep [m]
% stratColImpactIceN = zeros(numTimesteps,1); % ice from impacts for each timestep [m]
% totalIceN = zeros(numTimesteps,1); % total icer for each timestep with erosion [m]
% stratMatrixEjectaN = zeros(xyCells,xyCells,numTimesteps); % ejecta thickness for each location and timestep [m]
% totalEjectaMapN = zeros(xyCells,xyCells); % total ejecta thickness for each location
% ageMapN = ones(xyCells,xyCells)*10; % age of youngest crater for each location [Ga]

% Set erosion base
erosionBase = 0;

%% Main calculation
for t=1:numTimesteps
    
    % Finds age [Ga]
    age = round(timesteps(t),2);
    
    %% Ejecta
    % Select all craters that have this age
        whichCratersS = find(craterAgeS==age);
%     whichCratersN = find(craterAgeN==age);
    %
        if ~isempty(whichCratersS)
    
            % Cycle through each crater
            for i=1:length(whichCratersS)
    
                % Create map for just this crater's ejecta
                craterIdx = whichCratersS(i);
                tempEjectaMapS = ejS(:,:,craterIdx);
    
                % Add it to the ejecta matrix and total ejecta map
                totalEjectaMapS = totalEjectaMapS+tempEjectaMapS;
                stratMatrixEjectaS(:,:,t) = stratMatrixEjectaS(:,:,t) + tempEjectaMapS;
    
                % Create map with this crater's age
                R = craterDiamS(whichCratersS(i))/2*1000;
                tempAgeMap = mapAge(R,craterLatS(whichCratersS(i)),craterLonS(whichCratersS(i)),xyCells,xScale,'S',age);
    
                % Add to age map (keep youngest age for overlaps)
                ageMapS(tempAgeMap<ageMapS) = tempAgeMap(tempAgeMap<ageMapS);
    
            end
        end
    
%     if ~isempty(whichCratersN)
%         
%         % Cycle through each crater
%         for i=1:length(whichCratersN)
%             
%             % Create map for just this crater's ejecta
%             craterIdx = whichCratersN(i);
%             tempEjectaMapN = ejN(:,:,craterIdx);
%             
%             % Add it to the ejecta matrix and total ejecta map
%             totalEjectaMapN = totalEjectaMapN+tempEjectaMapN;
%             stratMatrixEjectaN(:,:,t) = stratMatrixEjectaN(:,:,t) + tempEjectaMapN;
%             
%             % Create map with this crater's age
%             R = craterDiamN(whichCratersN(i))/2*1000;
%             tempAgeMap = mapAge(R,craterLatN(whichCratersN(i)),craterLonN(whichCratersN(i)),xyCells,xScale,'N',age);
%             
%             % Add to age map (keep youngest age for overlaps)
%             ageMapN(tempAgeMap<ageMapN) = tempAgeMap(tempAgeMap<ageMapN);
%             
%         end
%     end
    
    %% Ice from volcanism
    if age >= 2.01 && age <=4
        
        if age >=3.01
                        iceMassVulxS = 1E7*.75*3000*(1000^3)*(10/1E6)*(1E7/1E9)*hopEfficS;
%             iceMassVulxN = 1E7*.75*3000*(1000^3)*(10/1E6)*(1E7/1E9)*hopEfficN;
        else
                        iceMassVulxS = 1E7*.25*3000*(1000^3)*(10/1E6)*(1E7/1E9)*hopEfficS;
%             iceMassVulxN = 1E7*.25*3000*(1000^3)*(10/1E6)*(1E7/1E9)*hopEfficN;
        end
        
        % Add ice to vector
                stratColVulxIceS(t) = iceMassVulxS;
%         stratColVulxIceN(t) = iceMassVulxN;
        
    else
                iceMassVulxS = 0;
%         iceMassVulxN = 0;
    end
    
    %% Ice from impact delivery
    
    % Calculate ice mass for this timestep
    [iceMassImpact] = impactIce(age,f24);
        iceMassImpactS = iceMassImpact*hopEfficS;
%     iceMassImpactN = iceMassImpact*hopEfficN;
    
    % Add ice to vector
        stratColImpactIceS(t) = iceMassImpactS;
%     stratColImpactIceN(t) = iceMassImpactN;
    
        totalIceMassS = iceMassImpactS+iceMassVulxS;
%     totalIceMassN = iceMassImpactN+iceMassVulxN;
    
    % Convert ice mass to volume
        totalIceVolS = totalIceMassS/934; % m^3
%     totalIceVolN = totalIceMassN/934; % m^3
    % Convert volume to thickness
        totalIceS(t) = totalIceVolS/((pctAreaS)*1000*1000); % m
%     totalIceN(t) = totalIceVolN/((pctAreaN)*1000*1000); % m
    
    if stratMatrixEjectaS(row,col,t) > 0.4
        erosionBase = t;
    end
    
    iceEroded = 0.1;
    layer = t;
    
    while iceEroded > 0
        
        if t > erosionBase
            
            if totalIceS(layer) >= iceEroded
                
                totalIceS(layer) = totalIceS(layer)-iceEroded;
                iceEroded = 0;
                
            else
                
                iceEroded = iceEroded-totalIceS(layer);
                totalIceS(layer) = 0;
                layer = layer-1;
                
            end
            
        else
            
            break
            
        end
        
    end
    
    
end

ageVecs = [craterAgeS];

end