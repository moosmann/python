data = sin(linspace(0,2*pi,100));
out = wavelet_transform(data);
reconvered = invwavelet_transform(out);

figure(1);
clf;
hold on;
plot(data);
plot(reconvered,'r')