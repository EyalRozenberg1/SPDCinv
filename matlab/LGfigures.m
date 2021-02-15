l = 0:1:9;
G2_sim = double(readNPY('C:\Users\eyalr\Documents\Projects\PyCharm\SPDCinv\SPDCinv\G2_reduced_2020-12-03_Nb500_Nx61Ny61_z0.001_steps100_loss_l1_N500_epochs500_qubit_n00n.npy'));
G2_sim = reshape(G2_sim,[10,10]);
G2_sim = abs(G2_sim)/sum(sum(abs(G2_sim)));
figure; imagesc(l,l,G2_sim); axis square; colorbar; title('simulation')

h=figure; b=bar3(G2_sim); 
xlabel('j (idler)'); ylabel('u (signal)'); zlabel('probability');
set(gca, 'XTickLabel', {'0', '1', '2', '3', '4', '5', '6', '7', '8','9'}, 'FontSize', 20, 'FontName', 'Calibri')
set(gca, 'YTickLabel',{'0', '1', '2', '3', '4', '5', '6', '7', '8','9'}, 'FontSize', 20, 'FontName', 'Calibri')
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

