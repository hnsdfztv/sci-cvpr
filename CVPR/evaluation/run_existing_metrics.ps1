Set-Location "d:\Zi_Liao\LEARING\DL\reproduce\dut\SCI\SCI\CVPR"

$out = "./results/metrics/current"

python evaluation/evaluate.py --input_dir ./data/additional --enhanced_dir ./results/additional_difficult_cpu --output_dir $out --tag additional_difficult_cpu
python evaluation/evaluate.py --input_dir ./data/additional --enhanced_dir ./results/additional_easy_cpu --output_dir $out --tag additional_easy_cpu
python evaluation/evaluate.py --input_dir ./data/additional --enhanced_dir ./results/additonal_medium_cpu --output_dir $out --tag additional_medium_cpu --allow_unpaired_fallback
python evaluation/evaluate.py --input_dir ./data/difficult --enhanced_dir ./results/difficult_cpu --output_dir $out --tag difficult_cpu
python evaluation/evaluate.py --input_dir ./data/easy --enhanced_dir ./results/easy_cpu --output_dir $out --tag easy_cpu
python evaluation/evaluate.py --input_dir ./data/medium --enhanced_dir ./results/medium_cpu --output_dir $out --tag medium_cpu

python evaluation/report_metrics.py --metrics_dir ./results/metrics --docs_dir ../../docs/evaluation --archive_legacy
