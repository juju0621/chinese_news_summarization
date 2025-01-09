python3 run_prediction.py \
        --test_file "${1}" \
        --model_name_or_path "./output_dir_adl_hw2"\
        --strategy "beam" \
        --num_beams 5 \
        --output_path "${2}"
