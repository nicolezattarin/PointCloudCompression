
for class_index in 1 100 7 8 9 10:
do
    root_dir="dataset_autoencoder_labels/label_"$class_index
    save_dir="dataset_autoencoder_labels/checkpoints_label_"$class_index
    python3.9 train.py --root $root_dir --batchsize 32 --epoches 100 --saved_path $save_dir
done