# requires gifsicle

# gifsicle -i results/maze.gif -O3 --colors 256 -o opt.gif

for file in *.gif; do
    gifsicle -i $file -O3 --colors 256 -o $file
done