# sh extract_mask.sh <data_dir> <prompt>
data_dir=$1
prompt=${2:-ceiling.wall.floor.poster.window}
python sam.py -i $data_dir/rgb -o $data_dir/mask --prompt $prompt