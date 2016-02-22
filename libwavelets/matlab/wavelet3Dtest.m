[X,Y,Z]=meshgrid(linspace(0,2*pi,100));
data = sin(X).*cos(Y).*atan(Z);
out = wavelet_transform(data);
reconvered = invwavelet_transform(out);

figure(1);
clf;
imagesc(data(:,:,end/2));

figure(2)
clf;
imagesc(reconvered(:,:,end/2))

[~,ind]=sort(-abs(out(:)));
out(ind(30:end))=0;
reconvered = invwavelet_transform(out);

figure(3)
clf;
imagesc(reconvered(:,:,end/2))