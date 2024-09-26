data_dir=$1
type=$2
task=$3
# if type = rectangle
if [ $type = rectangle ];
then
  python extract_mono_cues_rectangle.py --input_dir $data_dir/rgb --output_dir $data_dir --task $task
elif [ $type = square ];
then
  python extract_mono_cues_square.py --input_dir $data_dir/rgb --output_dir $data_dir --task $task
elif [ $type = highres ];
then
  python extract_mono_cues_highres.py --input_dir $data_dir/rgb --output_dir $data_dir --task $task
else
  echo "Invalid type"
fi