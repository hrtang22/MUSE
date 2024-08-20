DATA_PATH=[Your MSRVTT data and videos path]
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch --nproc_per_node=8 --master_port 6662 \
main_task_retrieval.py --do_train --num_thread_reader=16 \
--epochs=5 --batch_size=128 --n_display=50 \
--train_csv ${DATA_PATH}/anns/MSRVTT_train.9k.csv \
--val_csv ${DATA_PATH}/anns/MSRVTT_JSFUSION_test.csv \
--data_path ${DATA_PATH}/MSR-VTT/anns/MSRVTT_data.json \
--features_path ${DATA_PATH}/Compressed_videos \
--output_dir ckpts/ckpt_msrvtt_clip4clip_muse \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
--datatype msrvtt --expand_msrvtt_sentences  \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header MUSE \
--pretrained_clip_name ViT-B/32 \