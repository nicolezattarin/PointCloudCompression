for class_index in 1 100 7 8 9 10
do 
    checkpoint="dataset_autoencoder_labels/checkpoints_label_"$class_index"/checkpoints/test_min_loss.pth"
    save="results/label_"$class_index
    root_dir="dataset_autoencoder_labels/label_"$class_index
    python3.9 test.py --root $root_dir --checkpoint $checkpoint --save_dir $save
done