	.file	"spmv_csr.c"
  .section .rodata

  // pushq %rax
  // pushq %rcx
  // pushq %rdx
  // pushq %rsi
  // pushq %rdi
  // pushq %r8
  // pushq %r9
  // pushq %r10
  // pushq %r11
  // movl %r10d, %esi // number
  // movl %ecx, %edx // number
  // subl %r10d, %ecx // number
  // movl $0, %eax
  // leaq .LC0(%rip), %rdi // Empty string
  // call printf@PLT
  // popq %r11
  // popq %r10
  // popq %r9
  // popq %r8
  // popq %rdi
  // popq %rsi
  // popq %rdx
  // popq %rcx
  // popq %rax

.LC0:
  .string "rows_start is %d, rows_end is %d, len is %d, alpha is %f, beta is %f\n"
.LC00:
  .string "gather col_indx is %d,%d,%d,%d\n"
.LC01:
  .string "mov value is %f\n\n"
	.text
  .globl __spmv_csr_serial_host_avx2_double
	.type	__spmv_csr_serial_host_avx2_double, @function
#regester used :
#register unused: %r15
#void __spmv_csr_serial_host_avx2_double(
#                            const double alpha,   // xmm0
#                            const double beta,    // xmm1
#                            const int num_rows,         //rdi
#                            const int * Ap, //row_start //rsi
#                            const int * Aj, //col_idx   //rdx 
#                            const double * Ax, //value   //rcx
#                            const double * x,    //r8
#                                  double * y)    //r9
__spmv_csr_serial_host_avx2_double:
.LFB2:
	vmovups	.LC1(%rip), %ymm7  // Eight 0xffffffff

.LCFI0:
	pushq	%r14
	movq	%rdx, %r14 // r14 = col_indx
	pushq	%r13
	movq	%rcx, %r13 // r13 = value
	pushq	%r12
	movq	%r9, %r12 // r12 = y
	pushq	%rbp
	movl	%edi, %ebp #num_rows // ebp = num_rows
	pushq	%rbx
	movq	%rsi, %rbx  // rbx = rows_start
	xorl	%esi, %esi  // i 
	.p2align 4,,7

  movapd %xmm0, %xmm14 // alpha
  movapd %xmm1, %xmm15 // beta

.L17:
	cmpl	%ebp, %esi  // Same as sub %ebp, %esi (esi - ebp); have all rows been traversed?
	jge	.L10
	movslq	%esi,%rax            // i: row index
	leaq	0(,%rax,4), %rdx       // i*4
	movl	(%rbx,%rax,4), %r10d   // row_start[i]
	movl	4(%rbx,%rdx), %ecx     // row_start[i+1]
	leaq	(%r12,%rdx,2), %rdi      // y + i*8: addr
  xorps %xmm8,%xmm8
  xorps %xmm9,%xmm9
  xorps %xmm10,%xmm10
  subl  $3, %ecx                  // Unroll four: ecx = rows_end[i] - 3, unroll the loop 4 times
	movslq	%r10d,%rax              // row_start[i]

	leaq	(%r14,%rax,4), %rdx         // col_idx  addr
	leaq	(%r13,%rax,8), %r11         // value addr

	cmpl	%ecx, %r10d               // 
	jge	.L7
	.p2align 4,,7
// ########################main loop########################
.L6:
 // Unroll the loop 4 times, using two 128-bit fma operations
 // x -> xmm2
 // values -> xmm6
  movslq (%rdx),%rax          // col_indx[0]
  movsd  (%r8,%rax,8),%xmm2   // x[col_indx[0]]
  movslq 0x4(%rdx),%rax       // col_indx[1]
  movhpd (%r8,%rax,8),%xmm2
  movslq 0x8(%rdx),%rax       // col_indx[2]
  addl   $4,  %r10d           // r10d is the loop variable, incremented by 4 each time
  movupd  16(%r11), %xmm6       // value
  movupd  (%r11), %xmm5       // value
  movsd  (%r8,%rax,8),%xmm3   // x[col_indx[2]]
  movslq 0xc(%rdx),%rax       // col_indx[3]
  movhpd (%r8,%rax,8),%xmm3
  addq   $16, %rdx            // rdx is &col_idx[i], taking 4 elements each time

  // vinsertf128 $1, %xmm3, %ymm2, %ymm2 // Insert xmm3 into the high half of ymm2, result in ymm2; insert is too slow, pointless
	// vfmadd231pd %ymm6, %ymm2, %ymm8 // ymm8 is the intermediate accumulator; fma is slower than mul+add
  
  // Loop unrolling alone is enough
	vfmadd231pd	%xmm5, %xmm2, %xmm8 // ymm8 is the intermediate accumulator
	vfmadd231pd	%xmm6, %xmm3, %xmm9 // ymm9 is the intermediate accumulator
  addq  $32, %r11       // r11 is &value[i], +32 bytes each time
	cmpl	%r10d, %ecx     // ecx = nnz_row - 3
	jg	.L6
########################end loop########################
	// Reduction via extract is too slow, abandoned
  // vextractf128	$0x1, %ymm8, %xmm4
	addpd	%xmm8, %xmm9
  // Reduce xmm4 into xmm10
	vunpckhpd %xmm9,%xmm9,%xmm8
	addpd	%xmm9, %xmm8
.L7:
  subl %r10d, %ecx           // On exit, ecx + 3 - r10d is the number of remaining nnz, possible values 0,1,2,3; that is, ecx - r10d may be -3, -2, -1, 0
  cmpl	$-3, %ecx           
	jle	.L4                    // 0 left

	movslq	(%rdx),%rax        // col_idx[jj]
	movsd	(%r8,%rax,8), %xmm0  // x[] 
  vfmadd231sd (%r11), %xmm0, %xmm8 // val[0] * x[0] + accu

  cmpl	$-2, %ecx           
	jle	.L4                      // 1 left

  movslq	0x4(%rdx),%r9        // col_idx[jj]
	movsd	(%r8,%r9,8), %xmm2     // x[] 
  vfmadd231sd 0x8(%r11), %xmm2, %xmm8 // val[1] * x[1] + accu
	cmpl	$-1, %ecx               
	jle	.L4                      // 2 left

  movslq	0x8(%rdx),%rcx       // col_idx[jj]
	movsd	(%r8,%rcx,8), %xmm3    // x[] 
  vfmadd231sd 16(%r11), %xmm3, %xmm8 // val[2] * x[2] + accu

.L4:
  vmulsd (%rdi),%xmm15,%xmm1        // sum = y[i] * beta + 0 
	incl	%esi                        // Next row
  vfmadd231sd %xmm8, %xmm14, %xmm1 // y[i] = sum * alpha + y[i] 
	movsd	%xmm1, (%rdi)        // Write back y[i]
	jmp	.L17
.L10:
	popq	%rbx
	popq	%rbp
	popq	%r12
	popq	%r13
	popq	%r14
	ret
.LFE5279:
	.size	__spmv_csr_serial_host_avx2_double, .-__spmv_csr_serial_host_avx2_double
	.section	.rodata.cst32,"aM",@progbits,32
	.align 32
.LC1:
	.long	4294967295 # ffffffff
	.long	4294967295
	.long	4294967295
	.long	4294967295
	.long	4294967295
	.long	4294967295
	.long	4294967295
	.long	4294967295
	.ident	"GCC: (GNU) 9.3.1 20200408 (Red Hat 9.3.1-2)"
	.section	.note.GNU-stack,"",@progbits
