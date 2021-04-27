l = 0:1:9;
G2_sim = double(readNPY('G2 (29).npy'));
G2_sim = reshape(G2_sim,[9,9]);
G2_sim = abs(G2_sim)/sum(sum(abs(G2_sim)));
figure; imagesc(l,l,G2_sim); axis square; colorbar; title('simulation')

h=figure; b=bar3(G2_sim); 
zlim([0 0.22])
xlabel('j (idler)'); ylabel('u (signal)'); zlabel('probability');
title('Sim 30mm, MaxX=300, dx=8, dz=10, N=4000')
set(gca, 'XTickLabel', {'-4', '-3', '-2', '-1', '0', '1', '2', '3', '4','5'}, 'FontSize', 20, 'FontName', 'Calibri')
set(gca, 'YTickLabel',{'-4', '-3', '-2', '-1', '0', '1', '2', '3', '4','5'}, 'FontSize', 20, 'FontName', 'Calibri')
% colormap pink;
for k = 1:length(b)
    zdata = get(b(k),'ZData');
    b(k).CData = zdata;
    b(k).FaceColor = 'interp';
end


%%



% PolingCoeffs = readNPY('PlusMinus2Pair_L1mm_pump70um\PolingCoeffs.npy');
% %PolingCoeffs = reshape(PolingCoeffs, [5, 9]);
% 
% dx =1; %um
% MaxX = 120; %um 
% pump_waist = 70; %um 
% x = -MaxX:dx:MaxX;
% y = x;
% r0 = sqrt(2)*pump_waist;
% 
% 
% [X,Y] = meshgrid(x,y);
% Phi = atan2(Y,X);
% Rad = sqrt(X.^2+Y.^2)/r0;
% Profile = 0;
% alphas = [2.4048, 5.5201, 8.6537, 11.7915, 14.9309];
% 
% for p = 0:4
%     for ll = -4:4
%         Profile = Profile + PolingCoeffs(ll+5+9*p).*besselj(0, alphas(p+1)*Rad).*exp(-1j*ll*Phi);
%     end
% end
% Magnitude = abs(Profile)/max(max(abs(Profile)));
% phase = angle(Profile);
% dutycycle = asin(Magnitude)/pi;
% figure; imagesc(real(Profile)); colorbar
% 
% Z = -50:0.1:50; %um
% Lambda = 6.9328; %um
% DeltaK = 2*pi/Lambda;
% Poling = zeros(length(x),length(y), length(Z));
% for i = 1:length(Z)
%     i
%     z = Z(i);
%     for m = 0:100
%         if m == 0
%             Poling(:,:,i) = Poling(:,:,i) + 2*dutycycle - 1;
%         else
%             Poling(:,:,i) = Poling(:,:,i) + (2/(m*pi)).*sin(pi*m*dutycycle).*2.*cos(m*DeltaK*z + m * phase);
%         end
%     end
% end
% 
% %%
% figure; imagesc(sign(squeeze(Poling(:,:,502)))); colorbar
% figure; 
% subplot(3,1,1);
% imagesc(x,Z,sign(squeeze(Poling(:,round(end/3),:)))); ylabel('x[\mum]'); xlabel('z[\mum]'); text(-100,40,strcat('y=', num2str(y(round(end/3))),'um'),'Color','white','FontSize',16); set(gca, 'FontSize', 18); axis image
% subplot(3,1,2);
% imagesc(x,Z,sign(squeeze(Poling(:,round(end/2),:)))); ylabel('x[\mum]'); xlabel('z[\mum]'); text(-100,40,strcat('y=', num2str(y(round(end/2))),'um'),'Color','white','FontSize',16); set(gca, 'FontSize', 18); axis image
% subplot(3,1,3);
% imagesc(x,Z,sign(squeeze(Poling(:,round(2*end/3),:)))); ylabel('x[\mum]'); xlabel('z[\mum]'); text(-100,40,strcat('y=', num2str(y(round(2*end/3))),'um'),'Color','white','FontSize',16); set(gca, 'FontSize', 18); axis image

