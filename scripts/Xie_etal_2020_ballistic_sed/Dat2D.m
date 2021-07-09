function D_m=Dat2D(Dat_m)
%D_m,in meters,final diameter
%Dat_m,in meters,apparent diameter
if nargin<1
    close all
    Dat_m=logspace(3,5,1000);
end
D_m=Dat_m*1.43;
Dsc=19000;%m, Simple to complex transition ~19km (Pike 1980 LPSC, p27)
index=D_m>Dsc;
% D_m(index)=3.8*(Dat_m(index)/1000/2).^0.91*1000;
D_m(index)=1.52*Dsc^-0.18*Dat_m(index).^1.18;
if nargin<1
    loglog(Dat_m,D_m)
end