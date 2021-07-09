clear;close all

Coordinates_Apollo16=[15.5 -8.973];%LON,LAT  A16

R_Moon      = 1737.4;%km

step_Elevation=-0.05;
Elevation_from_PreSurface=(step_Elevation/2:step_Elevation:-1e4+step_Elevation/2)';%m

[NUM,~,RAW]=xlsread('BasinFormationSequence.xlsx');

[m_RAW,n_RAW]=size(RAW);
 
Coordinates_Basin=NUM(2:end,[4 3]);%LON,LAT
Rat=NUM(2:end,6)/2;%km
Name_basin=RAW(2:end,2);

R_A16=distance(Coordinates_Basin(:,2),Coordinates_Basin(:,1),Coordinates_Apollo16(2),Coordinates_Apollo16(1))/180*pi*R_Moon;%distance(LAT1,LON1,LAT2,LON2)
%---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
abundance_PEinED=[];%[Elevation_from_PreSurface, Fraction,SOI]
Elevation_from_PreSurface=-abs(Elevation_from_PreSurface);
FontSize=16;
C1={'r','c','b','k','m','g'};
for i=1:1:length(Rat)
    [Tlm(i),Tpe(i),abundance_PEinED]=Ballistic_Sedimentation_Model(Rat(i),R_A16(i),abundance_PEinED,Elevation_from_PreSurface);
    abundance_PEinED_withoutMixingbyLaterEjecta(:,i)=abundance_PEinED(:,i)*100;%
end
figure;set(gcf,'position',[400,160,800,600])%
for i=1:length(Rat)+1
    if i<=length(Rat)
        loglog(-Elevation_from_PreSurface+sum(Tpe(i+1:end)),abundance_PEinED_withoutMixingbyLaterEjecta(:,i),[C1{i} '--'],'LineWidth',2);hold on
        loglog(-Elevation_from_PreSurface,abundance_PEinED(:,i)*100,C1{i},'LineWidth',2);
        
        text(1800,7*2^(0.5*i),Name_basin{i},'Color',C1{i},'FontSize',FontSize,'FontName','Times New Roman')
    else 
        abundance_PreNectarianMaterials=100-abundance_PEinED_withoutMixingbyLaterEjecta(:,1);
        loglog(-Elevation_from_PreSurface+sum(Tpe(2:end)),abundance_PreNectarianMaterials,'--','Color',[0.5 0.5 0.5],'LineWidth',2);
        
        abundance_PreNectarianMaterials=(1-sum(abundance_PEinED(:,1:end),2))*100;
        loglog(-Elevation_from_PreSurface,abundance_PreNectarianMaterials,'Color',[0.5 0.5 0.5],'LineWidth',2);
        
        text(1800,5,sprintf('Pre-Nectarian\nmaterials'),'Color',[0.5 0.5 0.5],'FontSize',FontSize,'FontName','Times New Roman')
    end
end
set(gca,'FontSize',FontSize*0.75,'LineWidth',1);
ylim([0.1 100])
xlim([0.1 20000])

xlabel('Depth from surface (m)','FontSize',FontSize,'FontName','Times New Roman');
set(gca,'YTick',[1 2 5 10 20 50 100],'LineWidth',1.2,'TickLength',[0.015 0.01]);
ylabel('Abundance of basin ejecta in deposits (%)','FontSize',FontSize,'FontName','Times New Roman');





