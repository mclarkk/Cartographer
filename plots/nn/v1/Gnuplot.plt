set ylabel "Average Best-so-far"
set xlabel "Number of Births"
set xrange [0:10001]
set yrange [5:5000]
set xtics 0,2000,20000
set key top center horizontal
set logscale y

set output

set style line 1 lw 6

plot "ESS.txt" title "ES" with errorbars
replot "GAS.txt" title "GA" with errorbars
replot "EVS.txt" title "EV" with errorbars
replot "s-greedy.txt" title "Greedy" with errorbars
replot "s-rand.txt" title "Random" with errorbars

set terminal postscript color
set output "schwefelResults-e.ps"
replot