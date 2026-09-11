    .file	"ellmv_kernels.c"
    .text
    .globl	__spmv_sell_c8_serial_host_fma256_float
    .type	__spmv_sell_c8_serial_host_fma256_float, @function
__spmv_sell_c8_serial_host_fma256_float:
// void __spmv_sell_c8_serial_host_fma256_float(
// const float alpha:  xmm0
// const float beta :  xmm1
// const int num_chunks: edi
// const int *chunks_start: rsi
// const int *col_indices: rdx => rbx
// const float *values : rcx => r12
// const float *x :r8
// float *y : r9
// )
.L0:
    testl	%edi, %edi
    jle	.L11
    
    pushq	%r12
    pushq	%r13
    pushq	%r14
    pushq	%r15
    pushq	%rbp
    pushq	%rbx
    movq	%rdx, %rbx                // col_indices
    vbroadcastss	%xmm0, %ymm5 // alpha
    vbroadcastss	%xmm1, %ymm11 // beta
    vmovaps	%ymm11, %ymm1 // beta
    sall  $3, %edi                   // num_chunks * 8
    movl %edi, %ebp                  // Row count: num_chunks * 8
    xorl	%edx, %edx
  // outer loop on chunks,

  /*
rdx: outer loop variable, row index lrs, +8 per iteration
rbp: outer loop bound, row count num_chunks * 8
rdi: inner loop variable, address of values
r11: inner loop bound, address of the next column of values
  */
.L5:
    movq %rdx,%rax
    sarq	$3, %rax              // chunk_id
    movslq	(%rsi,%rax,4), %r11 // chunk_start
    movl	4(%rsi,%rax,4), %r10d // chunk_end
    movl	%r10d, %r12d       // chunk_end
    subl	%r11d, %r10d        // nnz_chunk
    sarl	$3, %r10d           // width = nnz_chunk / 8
    leaq	0(,%r11,4), %rdi    // chunk_start * 4
    leaq	(%rbx,%rdi), %rax   // &col_indices[chunk_start]
    testl	%r10d, %r10d        // Is width equal to 0?
    jle	.L6                   // No inner loop
    addq	%rcx, %rdi          // &values[chunk_start]
    leaq	(%rcx,%r12,4), %r11 // Address of the next column of values: values + chunk_end
    vxorps	%ymm7, %ymm7,%ymm7        // y[0:7]

  // inner loop on width
.L4:
    movslq	(%rax), %r10             // idx0
    movss	(%r8,%r10,4), %xmm0        // x[idx0]
    movslq	4(%rax), %r13            // idx1
    movss	(%r8,%r13,4), %xmm2        // x[idx1]
    shufps	$0, %xmm2, %xmm0
    movslq	8(%rax), %r14            // idx2
    movss	(%r8,%r14,4), %xmm4        // x[idx2]
    movslq	12(%rax), %r15           // idx3
    movss	(%r8,%r15,4), %xmm6        // x[idx3]
    shufps	$0, %xmm6, %xmm4
    shufps	$136, %xmm4, %xmm0       // x[idx0,idx1,idx2,idx3]

    movslq	16(%rax), %r10            // idx4
    movss	(%r8,%r10,4), %xmm8         // x[idx4]
    movslq	20(%rax), %r13            // idx5
    movss	(%r8,%r13,4), %xmm9         // x[idx5]
    shufps	$0, %xmm9, %xmm8          // x[idx4,idx5]
    movslq	24(%rax), %r14            // idx6
    movss	(%r8,%r14,4), %xmm10        // x[idx6]
    movslq	28(%rax), %r15            // idx7
    movss	(%r8,%r15,4), %xmm11        // x[idx7]
    shufps	$0, %xmm11, %xmm10
    shufps	$136, %xmm10, %xmm8      // x[idx4,idx5,idx6,idx7]

    VINSERTF128 $1,%xmm8,%ymm0,%ymm0
    vfmadd231ps	(%rdi), %ymm0, %ymm7 // acc += values[] * x[]

    addq	$32, %rdi                  // values, next column
    addq	$32, %rax                  // indices, next column
    cmpq	%rdi, %r11                 // rdi is the loop variable, r11 is the address of the next column of values
    jne	.L4
.L3:
    vmulps	(%r9,%rdx,4), %ymm1, %ymm0   // y[0-7] *= beta
    vfmadd132ps	%ymm5, %ymm0, %ymm7      // y[0-7] += alpha * acc
    vmovups	%ymm7, (%r9,%rdx,4)          // store(y[0-3])
    addq	$8, %rdx                       // Skip 8 rows
    cmpq	%rdx, %rbp                     // Finished?
    jne	.L5
    popq	%rbx
    popq	%rbp
    popq	%r15
    popq	%r14
    popq	%r13
    popq	%r12
    ret
.L6:                                  // Guard against the current row having no nnz
    vxorps	%ymm7, %ymm7, %ymm7
    jmp	.L3
.L11:
    ret


.LS:
  .string "rdi %ld , r11 %ld row id is %ld\n"

.ONE:
  .string "rows is  %ld \n"