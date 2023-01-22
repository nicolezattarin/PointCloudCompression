
for class_index in 1 7 8
do
    root_dir="dataset_autoencoder_labels/label_"$class_index
    save_dir="checkpoints/label_"$class_index
    python3.9 train.py --root $root_dir --batchsize 32 --epoches 40 --saved_path $save_dir --lr 0.0005
done