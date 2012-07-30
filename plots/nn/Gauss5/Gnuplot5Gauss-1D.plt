set title "Dim. 5, Gaussian, 500 points"
set ylabel "Average Proportion of Near Neighbors Conserved"
set xlabel "Neighborhood Radius"
set xrange [0.1:0.5]
set yrange [0.3:1]
set xtics 0.1
set ytics 0.1
set key bottom center horizontal

set output

set style line 1 lw 6 pt -1
set pointsize 3

plot "500-1D-hilbert.txt" title "Hilbert" with linespoints
replot "500-1D-zorder.txt" title "Z-order" with linespoints

set terminal postscript color
set output "nn-dim5-gauss-500-1D.ps"
replot
