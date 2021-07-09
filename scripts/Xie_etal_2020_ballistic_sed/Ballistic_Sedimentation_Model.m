function [T_LM_med,T_PE, abundance_PEinNewED, secondary_craters, Fraction_ExcavatedLM]=Ballistic_Sedimentation_Model(Rat,r_SOI_center,abundance_local_PE,Elevation_from_Local_Surface,T_Rat,bt,theta_degree,Y,K1,mu,nu)
%the code of the ballistic sedimentation model of Xie et al. (2020), created with R2016b.
%Xie, M., T. Liu, and A. Xu (2020), Ballistic sedimentation of impact crater ejecta: Implications for resurfacing and the provenance of lunar samples, Journal of Geophysical Research: Planets.

%-------------------------input parameters-------------------------------------------
%at least 4 input parameters.
%Rat: the apparent radius of a transient crater at preimpact surface in kilometers.

%r_SOI_center: the great-circle distance from SOI center to the parent crater center in kilometers.

%abundance_local_PE: the abundance of previously emplaced crater ejecta
%in eject deposits versus depth from the surface of the ejecta deposits.The
%ejecta deposits are the pre-existing local material for the emplacement of
%the ejecta of the current crater of interest; for the earliest crater, abundance_local_PE should be an empty array.

%Elevation_from_Local_Surface: depth from the surface of pre-existing local material. Note
%that local material often consists of earlier basin ejecta and pre-basin materials

%T_Rat: the thickness of ejecta at Rat with default value of T_Rat=0.068Rat.

%bt: the power-law index of ejecta thickness distribution with default value of bt=3.

%theta_degree: the launch/impact angle of ejecta in degrees with default value of theta_degree=45.

%Y: the strengh of target material.

%K1,mu and nu are the constants in crater scaling laws (Equation 12; see also Table 2).
%%-------------------------output parameters-----------------------------------------
%T_LM_med: the median thickness of excavated local material.

%T_PE: the thickness of primary ejecta.

%abundance_PEinNewED:the abundance of ejecta in ejecta depositsas a function of
%depth (i.e., Elevation_from_Local_Surface).

%secondary_craters: a struct recording the information of secondary craters.

%Fraction_ExcavatedLM:fraction of excavated local material as a function of
%depth (i.e., Elevation_from_Local_Surface).
%--------------------------------------------------------------------------

%--------------------------------------defaults--------------------------------------------------
if nargin<4%at least 4 input parameters.
    fprintf(2,'Error using Ballistic_Sedimentation_Model\nNot enough input arguments.\n\n');
    beep
    return;
end
if nargin<5||(nargin>=5&&isempty(T_Rat))
    T_Rat= 0.068*Rat;
end
if nargin<6||(nargin>=6&&isempty(bt))
    bt=3;
end
if nargin<7||(nargin>=7&&isempty(theta_degree))
    theta_degree = 45;
end


if nargin<8||(nargin>=8&&isempty(Y))
    Y= 10e6;%10 kPa
end
if nargin<9||(nargin>=9&&isempty(K1))
    K1=1.03;
end
if nargin<10||(nargin>=10&&isempty(nu))
    nu=0.4;
end
if nargin<11||(nargin>=11&&isempty(mu))
    mu=0.41;
end    
%-----------------------------------------------------------------------------------------------
Cex=3.5;%

Rm =1737.4;%km
rhot       = 3e6;% target density  g/m^3
rhoe       = rhot;% ejecta density  g/m^3
L_SOI      =2*r_SOI_center.^0.5;%km
launch_position=Rat;%km
S_SOI      = L_SOI.^2;%km^2
g          = 1.622;%k/s^2
g_km       = g/1000;%km/s^2
theta      = theta_degree*pi/180;
%--------------coefficients in Equation (10) derived from the work of Allen (1979).-------------   
a_rLSC=7.209219501948585;
b_rLSC=0.935213298326370;

Cmh=0.013;
b_MT=0.91;
%---------------------------------------  
N_SOI = length(r_SOI_center);%Number of SOI
r_SOI_outer    = r_SOI_center+L_SOI/2;%km, the outer boundary of the SOI
r_SOI_inner    = r_SOI_center-L_SOI/2;%km, the inner boundary of the SOI
% -------the area of a ring for spherical target in Equation (4)-------- %
S_ring   =  2*pi*Rm^2*(cos(r_SOI_inner/Rm)-cos(r_SOI_outer/Rm));% km2
S_ring   = abs(S_ring);%if r_SOI_outer/Rm>pi, S_ring<0
%-------------------------------------------------------------------------
if isempty(abundance_local_PE)
    m_depth=length(Elevation_from_Local_Surface);
    n_component =1;%the total number of ejecta sources.
    abundance_local_PE=zeros(m_depth,n_component ,N_SOI);%[depth,component,SOI]
    abundance_PEinNewED=zeros(size(abundance_local_PE));
%     IS_FIRST_BASIN=1;
else
    [~,n_component,k]=size(abundance_local_PE(:,:,1));%the total number of differently sourced ejecta in local material.
    n_component =n_component+1;%the total number of ejecta sources.%the total number of differently sourced ejecta in newly formed ejecta deposits.
    for i=1:k
        abundance_local_PE(:,n_component ,i)=0;%[depth,component,SOI]
%         IS_FIRST_BASIN=0;
    end
    abundance_PEinNewED=zeros(size(abundance_local_PE));
end
 
T_LM_med = zeros(size(r_SOI_center));%m
T_PE = zeros(size(r_SOI_center));%m
Fraction_ExcavatedLM=zeros(length(Elevation_from_Local_Surface),N_SOI);
for i_SOI =1:N_SOI
%     i_SOI
    if r_SOI_center(i_SOI)<=Rat+L_SOI(i_SOI)/2
        fprintf(2,'Error using Ballistic_Sedimentation_Model\nSOI has to be outside crater rim.\n\n');
        beep
        return;
    end
    Rs=@(r_gc)r_gc-launch_position;
    X=@(r_gc)Rs(r_gc)/(2*Rm);
    rgc2velocity=@(r_gc)sqrt(Rm*g_km*tan(X(r_gc))./(tan(X(r_gc))*cos(theta).^2 + sin(theta)*cos(theta)));

    v_SOI_inner  = rgc2velocity(r_SOI_inner(i_SOI));%km/s
    v_SOI_outer  = rgc2velocity(r_SOI_outer(i_SOI));%km/s
    v_SOI_center = rgc2velocity(r_SOI_center(i_SOI));%km/s

    if v_SOI_center>=2.38
        T_PE(i_SOI)  = 0;
        T_LM_med(i_SOI)=0;
        continue;
    end
    
    v2r_flat=@(v)launch_position + v.^2.*sin(2*theta)/(g_km);
    
    r_SOI_inner_flat     = v2r_flat(v_SOI_inner);%km
    r_SOI_outer_flat     = v2r_flat(v_SOI_outer);%km
    r_SOI_center_flat    = v2r_flat(v_SOI_center);%km
    
    S_ring_flat    = pi*(r_SOI_outer_flat^2 - r_SOI_inner_flat^2);%km^2
%------Equation (4)------------------------------
    Thickness_flat    = T_Rat*(r_SOI_center_flat./Rat).^-bt; %km,  the thickness of ejecta on flat surface at distance r_SOI_center_flat
    Thickness_SOI_km  = Thickness_flat*S_ring_flat/S_ring(i_SOI);  % km
    T_PE(i_SOI)       =Thickness_SOI_km*1000;%m
%-------Equation (5)-----------------------------
    M_SOI=rhoe*T_PE(i_SOI)*S_SOI(i_SOI)*1e6;%g
%------v_LSC,Equation (8)-----------------------
    r_LSC=a_rLSC*Rat^b_rLSC;
    v_LSC=rgc2velocity(r_LSC);%km/s
%-------equation (7)-----------------------------
    MT=0.09*pi*rhot*(Rat*1000)^3;%g
    if v_SOI_center>=v_LSC
        mh=Cmh*MT.^b_MT*(v_SOI_center/v_LSC).^(-5.7);%g, upper limit of ejecta mass
    else
        mh=Cmh*MT.^b_MT;%g, upper limit of ejecta mass
    end

    b=0.98;%

    ml=mh*1e-23;%lower limit of ejecta mass
    C_SOI=M_SOI*(1-b)/b/(mh.^(1-b)-ml^(1-b));% 2
    if C_SOI<0
        fprintf(2,'Error using Ballistic_Sedimentation_Model\nC_SOI<0\n\n');
        beep
        return
    end
%--------------------------cumulative M_SOI larger than or equal to M_SOI m  within SOI-----------
    N_mass_bin=round(log(mh/ml)/log(1.05))-1;%mh=ml*q^(n+1) => n=log(mh/ml)/log(q)-1,q=1.05
    mL=logspace(log10(ml),log10(mh),N_mass_bin+1);
    
    mR=mL(2:end);%right boundary of mass bin.
    mL=mL(1:end-1);%left boundary of mass bin.
    mass=(mL.*mR).^0.5;%g
    
    N_ejecta=C_SOI*mL.^-b-C_SOI*mR.^-b;%number of ejecta framents in each mass bin.
    %------------------scaling law--------------------------------------
    Vp=v_SOI_center*1000*sin(theta);%m/s, vertical component
    Vp2=Vp^2;
    ai=(3*mass/(4*pi*rhoe)).^(1/3);%m,radius of projectile
    Rat_sec=K1*ai.*((g*ai./Vp2)*(rhot/rhoe).^(2*nu/mu)+(Y./(rhot.*Vp2)).^(1+mu/2)*(rhot/rhoe).^(nu*(2+mu)/mu)).^(-mu/(2+mu));%m
    Dat=Rat_sec*2;
    %----------convert Dat to final diameter D (Equation (13))---------
    D=Dat2D(Dat);
    %-------------------------------------------------------------------
    d_ex  = 0.0134*Rat_sec*(v_SOI_center*1000)^0.38;   % Equation (16)


    T_onePE_Layer=max([Cex*d_ex(end)/5000,T_PE(i_SOI)/100]);%m
    N_layers=ceil(T_PE(i_SOI)/T_onePE_Layer);
    T_onePE_Layer=T_PE(i_SOI)/N_layers;
    
    elevation_bottom_layer=linspace(0,T_PE(i_SOI),N_layers+1);%the elevation of the lower boundary of a layer with respect to the surface of local material
    if length(elevation_bottom_layer)<=2
        elevation_bottom_layer=0;
    else
        elevation_bottom_layer(end)=[];
    end
    % ------------------------------------------------------local material (Equation 19)-----------------------------------------------------------------------------------------
    S_secondary_craters        = pi.*Rat_sec.^2;    % m^2, the area of secondary craters at surface
    density_secondary_craters    = N_ejecta./S_SOI(i_SOI)*1E-06;  %m^-2, density of secondary craters
    
    T_LM=zeros(1,N_mass_bin);
    d_eff=Cex.*d_ex;%effective excavation depth.
    E=zeros(N_mass_bin,N_layers);%the exponent in Equation (19)
    density_SCs_onelayer=density_secondary_craters/N_layers;%m^-2,the density of secondary craters formed by a layer of ejecta
    for k=1:N_layers
        for i=1:N_mass_bin
            dmin=d_ex(i);%the effective depth of the ith crater is dmin
            T_LM(i)=Cex.*dmin;
            
            index=i:N_mass_bin;
            index=index(T_LM(i)+elevation_bottom_layer(k)<d_eff(index));
            S_PIS=S_secondary_craters(index).*(1-T_LM(i)./d_eff(index)).*(1-elevation_bottom_layer(k)./d_eff(index));
            E(i,k)=E(i,k)+sum(S_PIS.*density_SCs_onelayer(index));
        end 
    end
    E=-cumsum(E,2);
    
    W_PIS_nLayers=1-exp(E);%Equation (19)
    W_PIS_allLayers=W_PIS_nLayers(:,end);
    

    T_LM_med(i_SOI)=T_LM(find(W_PIS_allLayers<0.5,1,'First'));
    if nargout<3
        continue;
    end
    Fraction_ExcavatedLM(:,i_SOI)=interp1(-T_LM,W_PIS_allLayers,Elevation_from_Local_Surface,'linear','extrap');%Figure 4a
    % ------------------------------------------------------Mixing-----------------------------------------------------------------------------------------
    Depth_Deposits=[elevation_bottom_layer(end:-1:1)'+T_onePE_Layer/2; Elevation_from_Local_Surface];
    %---------------One component Mixed by ejecta--------------
    d_mz=Cex*d_ex;
    dmz_max= -max(d_mz);%maximum depth of mixing zone
    dmz=(-T_onePE_Layer/2:-T_onePE_Layer:dmz_max)';
    step_dmz=dmz(1)-dmz(2);

    W_onelayer=W_PIS_nLayers(:,1);

    Wmz_onelayer=interp1(-d_mz,W_onelayer,dmz,'linear','extrap');
    Wmz_onelayer(Wmz_onelayer>1)=1;%correct inappropriate values derived from extrapolation if exist
    Wmz_onelayer(Wmz_onelayer<0)=0;%correct inappropriate values derived from extrapolation if exist

    T_ED_ex_by_onelayer=sum(step_dmz*Wmz_onelayer);

    Depth_Deposits_samebinsize=[flipud((step_dmz/2:step_dmz:T_PE(i_SOI))');dmz];%???????????????????????????
    binSize_one_ED_Layer=ones(size(Depth_Deposits_samebinsize))*step_dmz;

    abundance_PEinED_samebinsize=zeros(length(Depth_Deposits_samebinsize),n_component);
    for i=1:N_layers%mixing ejecta with excavated materials. Note that the excavated materials could be ejecta deposits formed by earlier ejecta and/or local material.
        if i==1
            index_PreED=find(Depth_Deposits_samebinsize>dmz_max+elevation_bottom_layer(i)&Depth_Deposits_samebinsize<elevation_bottom_layer(i));
            index_mz=1+floor((elevation_bottom_layer(i)-Depth_Deposits_samebinsize(index_PreED))/step_dmz);
            for j=1:n_component-1
                abundance_PEinED_samebinsize(index_PreED,j)=interp1(Elevation_from_Local_Surface,abundance_local_PE(:,j,i_SOI),Depth_Deposits_samebinsize(index_PreED),'linear','extrap');
%                     figure;semilogx(-Elevation_from_Local_Surface,abundance_local_PE(:,j,i_SOI),'r-',-Depth_Deposits_samebinsize(index_PreED),abundance_PEinED_samebinsize(index_PreED,j),'k--')
                abundance_PEinED_samebinsize(abundance_PEinED_samebinsize(:,j)>1,j)=1;%correct inappropriate values derived from extrapolation if exist
                abundance_PEinED_samebinsize(abundance_PEinED_samebinsize(:,j)<0,j)=0;%correct inappropriate values derived from extrapolation if exist
            end
        else
            index_PreED=index_PreED-1;
        end
        %-------------------------ejecta abundance---------------------------
        T_PE_in_exED=sum(binSize_one_ED_Layer(index_PreED).*abundance_PEinED_samebinsize(index_PreED,n_component).*Wmz_onelayer(index_mz));%ejecta in excavated ejecta deposits
        abundance_PE_in_excavatedMaterials=(T_onePE_Layer+T_PE_in_exED)/(T_ED_ex_by_onelayer+T_onePE_Layer);
        if 0%abundance_PE_in_excavatedMaterials>1
            sprintf(2,'Error using Ballistic_Sedimentation_Model\nabundance_PE_in_excavatedMaterials>1\n\n')
            beep;
            return
        end
        abundance_PEinED_samebinsize(index_PreED,n_component)=abundance_PEinED_samebinsize(index_PreED,n_component).*(1-Wmz_onelayer(index_mz))+Wmz_onelayer(index_mz)*abundance_PE_in_excavatedMaterials;%abundance_PEinED_samebinsize(index_PreED).*
        abundance_PEinED_samebinsize(index_PreED(1)-1,n_component)=abundance_PE_in_excavatedMaterials;
        for k=1:n_component-1%the jth component reworks preexisting materials, reducing the abundance of preexisting materials in ejecta deposits
            T_PE_in_exED=sum(binSize_one_ED_Layer(index_PreED).*abundance_PEinED_samebinsize(index_PreED,k).*Wmz_onelayer(index_mz));%ejecta in excavated ejecta deposits
            abundance_PE_in_excavatedMaterials=T_PE_in_exED/(T_ED_ex_by_onelayer+T_onePE_Layer);

            abundance_PEinED_samebinsize(index_PreED,k)=abundance_PEinED_samebinsize(index_PreED,k).*(1-Wmz_onelayer(index_mz))+Wmz_onelayer(index_mz)*abundance_PE_in_excavatedMaterials;%abundance_PEinED_samebinsize(index_PreED).*
            abundance_PEinED_samebinsize(index_PreED(1)-1,k)=abundance_PEinED_samebinsize(index_PreED(1)-1,k)+abundance_PE_in_excavatedMaterials;
        end
    end

    abundance_PEinNewED_oneSOI=zeros(length(Depth_Deposits),n_component);
    for j=1:n_component
        abundance_PEinNewED_oneSOI(:,j)=interp1(Depth_Deposits_samebinsize,abundance_PEinED_samebinsize(:,j),Depth_Deposits,'linear','extrap');
        abundance_PEinNewED_oneSOI(abundance_PEinNewED_oneSOI(:,j)>1,j)=1;%correct inappropriate values derived from extrapolation if exist
        abundance_PEinNewED_oneSOI(abundance_PEinNewED_oneSOI(:,j)<0,j)=0;%correct inappropriate values derived from extrapolation if exist
    end
    
    %---------------------------------------------------------------------------------------------------
    Fraction_PE_new1=[];
    for k=1:n_component  
        Fraction_PE_new1(:,k)=interp1(Depth_Deposits-T_PE(i_SOI),abundance_PEinNewED_oneSOI(:,k),Elevation_from_Local_Surface,'linear','extrap');
    end
    %--------------------------------------
    abundance_PEinNewED(:,:,i_SOI)=Fraction_PE_new1;

    if nargout>3
        secondary_craters(i_SOI).D=D;%m
        secondary_craters(i_SOI).largest=D(end);%m
        secondary_craters(i_SOI).S_SOI=S_SOI(i_SOI);% km^2
        secondary_craters(i_SOI).S_ring=S_ring(i_SOI);% km^2
        secondary_craters(i_SOI).density=N_ejecta/S_SOI(i_SOI);% km^-2
        secondary_craters(i_SOI).density_cumulative=fliplr(cumsum(secondary_craters(i_SOI).density(end:-1:1)));% km^-2
        secondary_craters(i_SOI).Num_SOI=N_ejecta;
        secondary_craters(i_SOI).Num_SOI_cumulative=fliplr(cumsum(secondary_craters(i_SOI).Num_SOI(end:-1:1)));
    end
end


