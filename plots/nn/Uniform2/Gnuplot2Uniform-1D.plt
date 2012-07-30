set title "Dim. 2, Uniform, 1000 points"
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

plot "1000-1D-hilbert.txt" title "Hilbert" with linespoints
replot "1000-1D-zorder.txt" title "Z-order" with linespoints

set terminal postscript color
set output "nn-dim2-uniform-1000-1D.ps"
replot
