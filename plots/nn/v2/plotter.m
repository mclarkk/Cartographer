load uniform_2_1000_h.ext
hx = uniform_2_1000_h(:,1);
hy = uniform_2_1000_h(:,2);
he = uniform_2_1000_h(:,3);

load uniform_2_1000_z.ext
zx = uniform_2_1000_z(:,1);
zy = uniform_2_1000_z(:,2);
ze = uniform_2_1000_z(:,3);

hold on

plot(hx, hy, '-b.')
%errorbar(hx,hy,he)
plot(zx, zy, '-r.')
%errorbar(zx,zy,ze)

axis([0,1,.4,1.02])
set(gca,'XTick',0:0.1:1)

xlabel('Radius')
ylabel('Proportion of near neighbors conserved')

title({'1-D Conservation of Near Neighbors versus Radius', 'Dim. 2, 1000 points, Uniform distribution'})

hLegend = legend('Hilbert', 'Z-order', 'Location', 'SouthEast');
hMarkers = findobj(hLegend,'type','line');
set(hMarkers(1), 'Color','red');
set(hMarkers(2), 'Color','red');

hold off
