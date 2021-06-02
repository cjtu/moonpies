function [totalIceMass] = impactIce(age,f24)

impactFlux = @(t) 3.76992E-13*(exp(6.93*t))+8.38E-4; % Derivative of eqn. X in Ivanov

%% Regime A: Micrometeoroids (<1 mm)
% Grun et al. 2011 give 10^6 kg per year of <1mm asteroid & comet grains

% Multiply by years per timestep and 10% hydration
totalImpactorWater = 1E6*1E7*.1;
% Scale for age
totalImpactorWater = totalImpactorWater*feval(impactFlux,age)/feval(impactFlux,0);
% Apply Ong et al. vapor retention and assume ballistic hopping
iceMassRegimeA = totalImpactorWater*.165;

%% Regime B: Small impactors (1 mm - 1 m)

impactorDiams = 0.01:0.0001:3; % in [m]
c = 1.568; % constant (Brown et al. 2002)
d = 2.7; % constant (Brown et al. 2002)
impactorNumGtLow = 10^(c-d*log10(impactorDiams(1))); % eqn. 3 (Brown et al. 2002) per year
impactorNumGtHigh = 10^(c-d*log10(impactorDiams(end))); % eqn. 3 (Brown et al. 2002) per year

% Number of impactors is (cumulative>lowest)-(cumulative>highest)
impactorNum = impactorNumGtLow-impactorNumGtHigh;
% Scale for length of timestep
impactorNum = impactorNum*(1E7);
% Scale for Earth/Moon ratio, using Mazrouei et al. 2019 supplemental
impactorNum = impactorNum/22.5;
% Scale for age
impactorNum = impactorNum*feval(impactFlux,age)/feval(impactFlux,0);

sfd = impactorDiams.^-3.7;
impactors = sfd*(impactorNum/sum(sfd));

% Calculate mass for each size bin
impactorMasses = 1300*(4/3)*pi*(impactorDiams/2).^3;
% Multiply masses by number in each bin
totalImpactorMass = sum(impactorMasses.*impactors);
% Averaged water contents using values from Jedicke et al. 2019
totalImpactorWater = totalImpactorMass*0.36*(2/3)*.1;

% Averaged retention from Ong et al., ballistic hopping
iceMassRegimeB = totalImpactorWater*0.165;

%% Regime C: Simple craters (steep branch)

craterDiams = 0.1:0.001:1.5; % in km
craterNum = neukum(craterDiams(1))-neukum(craterDiams(end));
craterNum = craterNum*(1E7/1E9); % scale for length of timestep
craterNum = craterNum*3.79E7; % scale for surface area of Moon

craterNum = craterNum*feval(impactFlux,age)/feval(impactFlux,0);

sfd = craterDiams.^-3.82; % Shallow branch on crater SFDs
craters = sfd*(craterNum/sum(sfd));

impactorDiams = dToL_C(craterDiams*1000,20);
impactorMasses = 1300*(4/3)*pi*(impactorDiams/2).^3;
totalImpactorMass = sum(impactorMasses.*craters);
totalImpactorWater = totalImpactorMass*0.36*(2/3)*.1;

iceMassRegimeC = totalImpactorWater*0.165;

%% Regime D: Simple craters (shallow branch)

craterDiams = 1.5:0.1:15; % in km
craterNum = neukum(craterDiams(1))-neukum(craterDiams(end));
craterNum = craterNum*(1E7/1E9); % scale for length of timestep
craterNum = craterNum*3.79E7; % scale for surface area of Moon

craterNum = craterNum*feval(impactFlux,age)/feval(impactFlux,0);

f = floor(craterNum);
if rand(1)<craterNum-f
    craterNum = f+1;
else
    craterNum = f;
end

probs = craterDiams.^-1.8; % Shallow branch on crater SFDs

craterPop = datasample(craterDiams,craterNum,'Replace',true,'Weights',probs);

craters = zeros(length(craterPop),1);

for i=1:length(craterPop)
    
    if rand(1)<0.36*(2/3) % hydrated carbonaceous
        craters(i) = craterPop(i);
    end
    
end

craters(craters==0) = [];

impactorSpeeds = normrnd(20,6,[length(craters),1]);
impactorSpeeds(impactorSpeeds<2.38) = 2.38; % minimum is Vesc
impactorDiams = dToL_D(craters*1000,impactorSpeeds);

impactorMasses = 1300*(4/3)*pi*(impactorDiams/2).^3;

waterRetained = zeros(length(impactorSpeeds),1);

for s=1:length(waterRetained)
    if impactorSpeeds(s) < 10
        waterRetained(s) = 0.5;
    else
        waterRetained(s) = 36.26*exp(-0.3464*impactorSpeeds(s));
    end
end

waterRetained(waterRetained<0) = 0;

waterMasses = impactorMasses.*waterRetained*0.1; % Assuming 10% hydration
iceMasses = zeros(length(waterMasses),1);

for w=1:length(waterMasses)
    
    iceMasses(w) = waterMasses(w);
    
end

iceMassRegimeD = sum(iceMasses);


%% Regime E: Complex craters (shallow branch)

craterDiams = 15:1:300; % in km
craterNum = neukum(craterDiams(1))-neukum(craterDiams(end));
craterNum = craterNum*(1E7/1E9); % scale for length of timestep
craterNum = craterNum*3.79E7; % scale for surface area of Moon

craterNum = craterNum*feval(impactFlux,age)/feval(impactFlux,0);

f = floor(craterNum);
if rand(1)<craterNum-f
    craterNum = f+1;
else
    craterNum = f;
end

probs = craterDiams.^-1.8; % Shallow branch on crater SFDs

craterPop = datasample(craterDiams,craterNum,'Replace',true,'Weights',probs);

craters = zeros(length(craterPop),1);

for i=1:length(craterPop)
    
    if rand(1)<0.36*(2/3) % hydrated carbonaceous
        craters(i) = craterPop(i);
    end
    
end

craters(craters==0) = [];

impactorSpeeds = normrnd(20,6,[length(craters),1]);
impactorSpeeds(impactorSpeeds<2.38) = 2.38; % minimum is Vesc
impactorDiams = dToL_E(craters*1000,impactorSpeeds,f24);

impactorMasses = 1300*(4/3)*pi*(impactorDiams/2).^3;

waterRetained = zeros(length(impactorSpeeds),1);

for s=1:length(waterRetained)
    if impactorSpeeds(s) < 10
        waterRetained(s) = 0.5;
    else
        waterRetained(s) = 36.26*exp(-0.3464*impactorSpeeds(s));
    end
end

waterRetained(waterRetained<0) = 0;

waterMasses = impactorMasses.*waterRetained*0.1; % Assuming 10% hydration
iceMasses = zeros(length(waterMasses),1);

for w=1:length(waterMasses)
    
    iceMasses(w) = waterMasses(w);
    
end

iceMassRegimeE = sum(iceMasses);

totalIceMass = iceMassRegimeA+iceMassRegimeB+iceMassRegimeC+iceMassRegimeD+iceMassRegimeE;

end