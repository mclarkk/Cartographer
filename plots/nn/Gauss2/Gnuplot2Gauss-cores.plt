set xlabel "Radius"
set ylabel "Cores"
set zlabel "Avg. Cs per N"
set xrange [0.1:0.5]
set yrange [100:500]
set zrange [5:460]
set xtics 0.1
set ytics 100
set ztics 0,50
set key top center horizontal

set output

set style line 1 lw 6 pt -1
set pointsize 3
set dgrid3d 3,5
#set pm3d

splot "1000-cores-hilbert.txt" title "Hilbert" with linespoints
replot "1000-cores-zorder.txt" title "Z-Order" with linespoints

set terminal postscript color 
set output "nn-dim2-gauss-1000-cores.ps"
replot
