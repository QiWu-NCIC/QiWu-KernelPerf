    .file	"ellmv_kernels.c"
    .text
    .globl	__spmv_sell_c4_serial_host_fma128_double
    .type	__spmv_sell_c4_serial_host_fma128_double, @function
__spmv_sell_c4_serial_host_fma128_double:
// void __spmv_sell_c4_serial_host_fma128_double(
// const double alpha:  xmm0
// const double beta :  xmm1
// const int num_chunks: edi
// const int *chunks_start: rsi
// const int *col_indices: rdx => rbx
// const double *values : rcx => r12
// const double *x :r8
// double *y : r9
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
    vbroadcastsd	%xmm0, %ymm5 // alpha
    vbroadcastsd	%xmm1, %ymm11 // beta
    vmovapd	%ymm11, %ymm1 // beta
    sall  $2, %edi                   // num_chunks * 4
    movl %edi, %ebp                  // Row count: num_chunks * 4
    xorl	%edx, %edx                 // Row count
    xorq	%r15, %r15                 // chunk_id = row / 4
  // outer loop on chunks,

  /*
rdx: outer loop variable, row index lrs, +4 per iteration
rbp: outer loop bound, row count num_chunks * 4
rdi: inner loop variable, address of values
r11: inner loop bound, address of the next column of values
  */
.L5:
    movslq	(%rsi,%r15,4), %r13   // chunk_start
    movslq	4(%rsi,%r15,4), %r10  // chunk_end
    leaq	(%rcx,%r13,8), %rdi     // &values[chunk_start]
    leaq	(%rbx,%r13,4), %rax     // &col_indices[chunk_start]

    leaq	(%rcx,%r10,8), %r11   // &values[chunk_end]
    cmpl	%r13d, %r10d          // Is width equal to 0?
    jle	.L6                     // No inner loop
    xorpd	%xmm7, %xmm7        // y[0:1]
    xorpd	%xmm8, %xmm8        // y[2:3]
  // inner loop on width
.L4:
    movslq	(%rax), %r10             // idx0
    movsd	(%r8,%r10,8), %xmm0        // x[idx0]
    movslq	4(%rax), %r13            // idx1
    movhpd	(%r8,%r13,8), %xmm0        // x[idx1,idx0]
    movslq	8(%rax), %r14            // idx2
    movsd	(%r8,%r14,8), %xmm4        // x[idx2]
    movslq	12(%rax), %r12           // idx3
    movhpd	(%r8,%r12,8), %xmm4        // x[idx3,idx2]

    vfmadd231pd	(%rdi), %xmm0, %xmm7 // acc += values[] * x[]
    vfmadd231pd	16(%rdi), %xmm4, %xmm8 // acc += values[] * x[]

    addq	$32, %rdi                  // values, next column
    addq	$16, %rax                  // indices, next column
    cmpq	%rdi, %r11                 // rdi is the loop variable, r11 is the address of the next column of values
    jne	.L4
.L3:
    vmulpd	(%r9,%rdx,8), %xmm1, %xmm2   // y[0-1] *= beta
    vmulpd	16(%r9,%rdx,8), %xmm1, %xmm3   // y[2-3] *= beta
    vfmadd132pd	%xmm5, %xmm2, %xmm7      // y[0-1] += alpha * acc
    vfmadd132pd	%xmm5, %xmm3, %xmm8      // y[2-3] += alpha * acc
    movupd	%xmm7, (%r9,%rdx,8)          // store(y[0-1])
    movupd	%xmm8, 16(%r9,%rdx,8)          // store(y[2-3])
    incq	%r15                           // chunk_id++
    addq	$4, %rdx                       // Skip 4 rows
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
    xorpd	%xmm7, %xmm7
    xorpd	%xmm8, %xmm8
    jmp	.L3
.L11:
    ret


.LS:
  .string "rdi %ld , r11 %ld row id is %ld\n"

.ONE:
  .string "rows is  %ld \n"