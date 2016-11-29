analysis_dir='../results'
mkdir $analysis_dir
mkdir $analysis_dir/figs
mkdir $analysis_dir/tables
mkdir ../results/figs/gender
mkdir ../results/figs/experience
mkdir ../results/figs/national
mkdir ../results/figs/champ
mkdir ../results/figs/repeat
mkdir ../results/figs/cm
mkdir ../results/figs/height
mkdir ../results/figs/all
mkdir ../results/figs/topline


python champ.py > $analysis_dir/tables/results.txt
grep -E '^\d|^[A-Z]|All' $analysis_dir/tables/results.txt > $analysis_dir/tables/toplines.txt
grep -E '^\d|^[A-Z]|\*|All' $analysis_dir/tables/results.txt > $analysis_dir/tables/significant.txt
python anthropometry.py
