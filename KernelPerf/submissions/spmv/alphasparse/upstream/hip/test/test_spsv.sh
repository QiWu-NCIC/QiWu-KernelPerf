# Test matrix set
# Create two folders named metrics and results in the working directory first
# The measurements produced by the selected algorithm alg_num are stored as CSV files under ./results/
# Each row of data is: [matrix_name, hip_time, alpha_time, speedup]

# alg_num=1: capellini-spsv
# alg_num=2: cublk
# alg_num=3: nnz-balance

# With alg_num=1, non-transposed & transposed, unit & non-unit diagonal, lower & upper triangular, and f64 & f32 have all been tested and produce correct results
# With alg_num=2 or 3 the code is experimental and supports CSR, non-transposed, non-unit diagonal, lower triangular, f64 & f32

if [ -f './metrics.txt' ]; then
    rm ./metrics.txt
fi

function get_files {
    local l_files=$(find /public/home/guochengxin_ict/zk/matrix/ -name "*.mtx")
    echo ${l_files[@]}
}

function get_iter_warmup {
    local ret=()
    local l_iter=1
    local l_warmup=0
    ret+=(${l_iter})
    ret+=(${l_warmup})
    echo "${ret[@]}"
}


res=($(get_files))
# File path list in ascending order
files=${res[@]}

# File path list in descending order
# files=($(echo ${res[@]} | tac -s ' '))
# files=${files[@]}

res=($(get_iter_warmup))
iter=${res[0]}
warmup=${res[1]}

TIMEOUT=180

date_token=`date +"%m%d%H%M"`
new_filename_token='spsv_test.csv'                    
speedup_file='./results/'${date_token}'_results_'${new_filename_token}
metrics_file='./metrics/'${date_token}'_metrics_'${new_filename_token}    
echo ${speedup_file}
echo ${metrics_file}
# Add the header row
echo "mtx,hip,alpha,speedup," | tee -a ${speedup_file}
for file in ${files}; do
	cur_mat_name=$(basename ${file})
	echo -n ${cur_mat_name}"," | tee -a ${speedup_file}
	{
		timeout --preserve-status --signal=SIGTERM ${TIMEOUT} /public/home/guochengxin_ict/gcx/alphasparse_for_test/build/hip/test/spsv_csr_r_f64_test_metrics \
			--data-file=${file} \
			--diagA=N --fillA=L --transA=N --alg_num=2 \
			--iter=${iter} --warmup=${warmup} \
			--metrics # \
			# --check
			if [ $? -ne 0 ]; then
			echo -n "0,"
		fi
	} | tee -a ${speedup_file}
	echo "" | tee -a ${speedup_file}
done
if [ -f './metrics.txt' ]; then
	mv ./metrics.txt ${metrics_file}
fi

# Remove invalid data rows
awk -F',' 'NF > 2 && $0 !~ /,0,/' ${speedup_file} > "${speedup_file}.tmp"
mv "${speedup_file}.tmp" "${speedup_file}"
