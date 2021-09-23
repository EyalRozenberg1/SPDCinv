l = -4:1:4;

folder = 'C:\Users\avivk\OneDrive\Desktop\Aviv\SPDC\Machine Learning\Figures for the paper\ququart2_only_crystal';

G2_sim = double(readNPY(strcat(folder,'\coincidence_rate_PumpLG0m2.npy')));
G2_sim = reshape(G2_sim,[9,9]);
G2_sim = abs(G2_sim)/sum(sum(abs(G2_sim)));
% figure; imagesc(l,l,G2_sim); axis square; colorbar; title('simulation'); set(gca, 'FontSize', 20, 'FontName', 'Calibri'); xlabel('l_i (idler)'); ylabel('l_s (signal)');

G2_target = double(readNPY(strcat(folder,'\target.npy')));
G2_target = reshape(G2_target,[9,9]);
G2_target = abs(G2_target)/sum(sum(abs(G2_target)));


h=figure; b=bar3(G2_sim); 
zlabel('probability');
set(gca, 'XTickLabel', {'-4', '-3', '-2', '-1', '0', '1', '2', '3', '4'}, 'FontSize', 28, 'FontName', 'Calibri')
set(gca, 'YTickLabel',{'-4', '-3', '-2', '-1', '0', '1', '2', '3', '4'}, 'FontSize', 28, 'FontName', 'Calibri')
colormap summer;
for k = 1:length(b)
    zdata = get(b(k),'ZData');
    b(k).CData = zdata;
    b(k).FaceColor = 'interp';
end
zlim([0, 0.5]);


h=figure; b=bar3(G2_target); 
zlabel('probability');
set(gca, 'XTickLabel', {'-4', '-3', '-2', '-1', '0', '1', '2', '3', '4'}, 'FontSize', 28, 'FontName', 'Calibri')
set(gca, 'YTickLabel',{'-4', '-3', '-2', '-1', '0', '1', '2', '3', '4'}, 'FontSize', 28, 'FontName', 'Calibri')
colormap summer;
for k = 1:length(b)
    zdata = get(b(k),'ZData');
    b(k).CData = zdata;
    b(k).FaceColor = 'interp';
end
zlim([0, 0.5]);





%%

dx = 1e-6; %m
MaxX = 120e-6; %m 
%pump_waist = 40e-6; %m 
x = -MaxX:dx:MaxX;
y = x;

Pump = 1;

if Pump
    pump_coeffs_real = [0 0 1 0 0 0 0 0 0]; %readNPY(strcat(folder,'\parameters_pump_real.npy'));
    pump_coeffs_imag = [0 0 0 0 0 0 0 0 0];%readNPY(strcat(folder,'\parameters_pump_imag.npy'));
    pump_coeffs = pump_coeffs_real + 1i*pump_coeffs_imag;
    pump_waists_vec = 1e-6*[40 40 40 40 40 40 40 40 40]; %readNPY(strcat(folder,'\parameters_pump_waists.npy'));
    %*34/40

    MaxP = 0;
    MaxL = 4;%2;
  
    PumpCoeffs = pump_coeffs;

    [X,Y] = meshgrid(x,y);
    PumpProfile = 0;

    for p = 0:MaxP
        for ll = -MaxL:MaxL
            pump_waist = pump_waists_vec(ll+MaxL+1+(2*MaxL+1)*p);
            PumpProfile = PumpProfile + PumpCoeffs(ll+MaxL+1+(2*MaxL+1)*p).*LaguerreGauss(404e-9, 2, pump_waist,ll,p,X,Y,0);
        end
    end
        
    figure; imagesc(x*1e6, y*1e6, abs(PumpProfile)); axis square; colorbar; set(gca, 'FontSize', 28, 'FontName', 'Calibri'); xlabel('x[um]'); ylabel('y[um]');
    axes('Position',[.53 .7 .2 .2])
    box on
    imagesc(x*1e6, y*1e6, angle(PumpProfile)); axis square; colorbar; set(gca, 'FontSize', 16, 'FontName', 'Calibri', 'XColor', 'w', 'YColor', 'w'); xlabel('x[um]', 'Color', 'w'); ylabel('y[um]', 'Color', 'w');
    c = colorbar; c.Color = 'w';
end

%%


Poling = 1;


if Poling
    poling_coeffs_real = readNPY(strcat(folder,'\parameters_crystal_real.npy'));
    poling_coeffs_imag = readNPY(strcat(folder,'\parameters_crystal_imag.npy'));
    poling_coeffs = poling_coeffs_real + 1i*poling_coeffs_imag;
    poling_waist = 1e-5*readNPY(strcat(folder,'\parameters_crystal_effective_waists.npy'));

%     poling_coeffs_real = zeros(size(poling_coeffs_real));
%     poling_coeffs_imag = poling_coeffs_real;
% 
%     poling_coeffs_real(1+2+1+(2*2+1)*0) = 1; 
%     
%     poling_coeffs = poling_coeffs_real + 1i*poling_coeffs_imag;
%     poling_waist = 40e-6*ones(size(poling_waist));
    
    
    MaxP = 2;%9;
    MaxL = 2;%4;
  
    PolingCoeffs = poling_coeffs;

    [X,Y] = meshgrid(x,y);
    Profile = 0;

    for p = 0:MaxP
        for ll = -MaxL:MaxL
            waist_curr = poling_waist(ll+MaxL+1+(2*MaxL+1)*p);
            Profile = Profile + PolingCoeffs(ll+MaxL+1+(2*MaxL+1)*p).*LaguerreGauss(808e-9, 2, waist_curr,ll,p,X,Y,0);
        end
    end
    
    Profile = Profile / max(max(abs(Profile)));
    
    Magnitude = abs(Profile);
    phase = angle(Profile);
    dutycycle = asin(Magnitude)/pi;
%     figure; imagesc(real(Profile)); title('Real(poling)'); colorbar
%     figure; imagesc(imag(Profile)); title('Imag(poling)'); colorbar
%     figure; imagesc(Magnitude); title('abs(poling)'); colorbar
%     figure; imagesc(phase); title('phase(poling)'); colorbar

%%
    Lambda = 9.87; %um

    Z = -3*Lambda/2:1:1.1*3*Lambda/2; %um

    
    DeltaK = 2*pi/Lambda;
    Poling = zeros(length(x),length(y), length(Z));
    for i = 1:length(Z)
        100*i/length(Z)
        z = Z(i);
        for m = 0:100
            if m == 0
                Poling(:,:,i) = Poling(:,:,i) + 2*dutycycle - 1;
            else
                Poling(:,:,i) = Poling(:,:,i) + (2/(m*pi)).*sin(pi*m*dutycycle).*2.*cos(m*DeltaK*z + m * phase);
            end
        end
    end
end


%%

Draw3DPoling = 0;

if Draw3DPoling
    s_pol = size(Poling);
    z_factor = 20;
    figure; axis image; axis off; 
    for ii = 1:s_pol(1)
        100*ii/s_pol(1)
        for jj = 1:s_pol(2)
            for kk = 1:s_pol(3)
                if sign(Poling(ii,jj,kk)) == 1
                    plotcube([1 z_factor 1],[ii  z_factor*(kk-1) + 1  jj],1,[102, 163, 255]/256);
                    hold on;
                end
            end
        end
    end

    plotcube([241 length(Z)*z_factor 241],[1  1  1],.1,[17, 103, 177]/256);


    view(130,20);
    camlight('right')
end

