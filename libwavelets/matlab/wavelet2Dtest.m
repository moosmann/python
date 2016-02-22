[X,Y]=meshgrid(linspace(0,2*pi,100));
data = sin(X).*cos(Y);
out = wavelet_transform(data);
reconvered = invwavelet_transform(out);

figure(1);
clf;
imagesc(data);

figure(2)
clf;
imagesc(reconvered)

[~,ind]=sort(-abs(out(:)));
out(ind(30:end))=0;
reconvered = invwavelet_transform(out);

figure(3)
clf;
imagesc(reconvered)