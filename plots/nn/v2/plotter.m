load gauss_2_1000_h.ext
hx = gauss_2_1000_h(:,1);
hy = gauss_2_1000_h(:,2);
he = gauss_2_1000_h(:,3);

load gauss_2_1000_z.ext
zx = gauss_2_1000_z(:,1);
zy = gauss_2_1000_z(:,2);
ze = gauss_2_1000_z(:,3);

hold on

plot(hx, hy, '-b.')
%errorbar(hx,hy,he)
plot(zx, zy, '-r.')
%errorbar(zx,zy,ze)

axis([0,1,.4,1.02])
set(gca,'XTick',0:0.1:1)

xlabel('Radius')
ylabel('Proportion of near neighbors conserved')

title({'1-D Conservation of Near Neighbors versus Radius', 'Dim. 2, 1000 points, Gaussian distribution'})

hLegend = legend('Hilbert', 'Z-order', 'Location', 'SouthEast');
hMarkers = findobj(hLegend,'type','line');
set(hMarkers(1), 'Color','red');
set(hMarkers(2), 'Color','red');

hold off
