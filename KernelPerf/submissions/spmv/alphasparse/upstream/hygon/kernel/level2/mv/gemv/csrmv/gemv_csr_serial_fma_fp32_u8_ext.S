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
  // movl %ecx, %esi // number
  // movl $0, %eax
  // leaq .Llast(%rip), %rdi // Empty string
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
  .string "mov col_indx is %d\n\n"
.Llast:
  .string "leftover is %d\n\n"

	.text
  .globl __spmv_csr_serial_host_avx2_float
	.type	__spmv_csr_serial_host_avx2_float, @function
#regester used :
#register unused: %r15
#void __spmv_csr_serial_host_avx2_float(
#                            const float alpha,   // xmm0
#                            const float beta,    // xmm1
#                            const int num_rows,         //rdi
#                            const int * Ap, //row_start //rsi
#                            const int * Aj, //col_idx   //rdx 
#                            const float * Ax, //value   //rcx
#                            const float * x,    //r8
#                                  float * y)    //r9
# parameters:
__spmv_csr_serial_host_avx2_float:
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

  vmovaps %ymm0, %ymm14 // alpha
  vmovaps %ymm1, %ymm15 // beta

.L17:
	cmpl	%ebp, %esi  // Same as sub %ebp, %esi (esi - ebp)
	jge	.L10
	movslq	%esi,%rax            // i: row index
	leaq	0(,%rax,4), %rdx       //i*4
	movl	(%rbx,%rax,4), %r10d   // row_start[i]
	movl	4(%rbx,%rdx), %ecx     // row_start[i+1]
	leaq	(%r12,%rdx), %rdi      // y[i] addr
  vxorps %ymm8,%ymm8,%ymm8
  vxorps %ymm10,%ymm10,%ymm10

  subl  $7, %ecx                  // unroll four, ecx = rows_end[i] - 7
	movslq	%r10d,%rax              // row_start[i]
	salq	$2, %rax                  // addr of row_start

	leaq	(%rax,%r14), %rdx         // col_idx  addr
	leaq	(%rax,%r13), %r11         // value addr

	cmpl	%ecx, %r10d               // 
	jge	.L7
	.p2align 4,,7


########################main loop########################
.L6:
 // x -> xmm2
 // values -> xmm6
  movslq (%rdx),%rax          // col_indx[0]
  movss  (%r8,%rax,4),%xmm2   // x[col_indx[0]]
  movslq 0x4(%rdx),%rax       // col_indx[1]
  movss  (%r8,%rax,4),%xmm0   // x[col_indx[1]]
  movslq 0x8(%rdx),%rax       // col_indx[2]
  shufps $0x0,%xmm0,%xmm2    
  addl   $8,  %r10d           // r10d is the loop variable, incremented by 8 each time
  vmovups  (%r11), %ymm6       // value
  movss  (%r8,%rax,4),%xmm3   // x[col_indx[2]]
  movslq 0xc(%rdx),%rax       // col_indx[3]
  movss  (%r8,%rax,4),%xmm4   // x[col_indx[3]]
  shufps $0x0,%xmm4,%xmm3 
  shufps $0x88,%xmm3,%xmm2
  vmovaps %xmm2,%xmm11       // xmm11 -> x[0:3]

  movslq 16(%rdx),%rax          // col_indx[4]
  movss  (%r8,%rax,4),%xmm2   // x[col_indx[4]]
  movslq 20(%rdx),%rax       // col_indx[5]
  movss  (%r8,%rax,4),%xmm0   // x[col_indx[5]]
  movslq 24(%rdx),%rax       // col_indx[6]
  shufps $0x0,%xmm0,%xmm2    
  movss  (%r8,%rax,4),%xmm3   // x[col_indx[6]]
  movslq 28(%rdx),%rax       // col_indx[7]
  movss  (%r8,%rax,4),%xmm4   // x[col_indx[7]]
  shufps $0x0,%xmm4,%xmm3 
  shufps $0x88,%xmm3,%xmm2
  addq   $32, %rdx            // rdx is &col_idx[i], +32 each time

  vinsertf128 $1, %xmm2, %ymm11, %ymm2 // Insert xmm2 into the high half of ymm11, with the result in ymm2
  
  // vmovaps %ymm7, %ymm3
  // vmovdqu  (%rdx),%ymm5     // col_indx[0,1,2,3]
  // addq $32, %rdx // rdx is &col_idx[i], +32 each time
  // addl $8, %r10d // r10d is the loop variable, +8 each time
  // vgatherdps	%ymm3, (%r8,%ymm5,4), %ymm2
  
	vfmadd231ps	%ymm6, %ymm2, %ymm8 // ymm8 is the intermediate accumulator; fma is slower than mul+add
  addq  $32, %r11       // r11 is &value[i], +32 each time
	cmpl	%r10d, %ecx     // ecx = nnz_row - 7
	jg	.L6
########################end loop########################
	vextractf128	$0x1, %ymm8, %xmm4
	vaddps	%xmm8, %xmm4, %xmm4
  // Reduce xmm4 into xmm10
	vmovhlps	%xmm4, %xmm4, %xmm8
	vaddps	%xmm4, %xmm8, %xmm4
	vshufps	$85, %xmm4, %xmm4, %xmm8
	vaddps	%xmm4, %xmm8, %xmm10
.L7:
  addl  $7, %ecx // ecx now holds rows_end[i]
.L8:
  // If the computation is not finished, jump to .L4; r10d holds the number of nz already traversed
	cmpl	%r10d, %ecx          // If rows_end[i] equals rows_start[i], jump away, otherwise execute .L9
  jle .L4
.L9:
  subl %r10d, %ecx           // How many are left?
	movslq	(%rdx),%rax        // col_idx[jj]
	movss	(%r8,%rax,4), %xmm0  // x[] 
	mulss	(%r11), %xmm0        // value
	addss	%xmm0, %xmm10        // xmm10 = sum
	cmpl	$1, %ecx           
	je	.L4                    // 1 left

  movslq	0x4(%rdx),%rax        // col_idx[jj]
	movss	(%r8,%rax,4), %xmm0     // x[] 
	mulss	0x4(%r11), %xmm0        // value
	addss	%xmm0, %xmm10           // xmm10 = sum
	cmpl	$2, %ecx               
	je	.L4                       // 2 left

  movslq	0x8(%rdx),%rax        // col_idx[jj]
	movss	(%r8,%rax,4), %xmm0     // x[] 
	mulss	0x8(%r11), %xmm0        // value
	addss	%xmm0, %xmm10           // xmm10 = sum
	cmpl	$3, %ecx              
	je	.L4                       // 3 left

  movslq 12(%rdx),%rax        // col_idx[jj]
	movss	(%r8,%rax,4), %xmm0     // x[] 
	mulss	12(%r11), %xmm0        // value
	addss	%xmm0, %xmm10           // xmm10 = sum
	cmpl	$4, %ecx              
	je	.L4                       // 4 left

  movslq 16(%rdx),%rax        // col_idx[jj]
	movss	(%r8,%rax,4), %xmm0     // x[] 
	mulss	16(%r11), %xmm0        // value
	addss	%xmm0, %xmm10           // xmm10 = sum
	cmpl	$5, %ecx              
	je	.L4                       // 5 left

  movslq 20(%rdx),%rax        // col_idx[jj]
	movss	(%r8,%rax,4), %xmm0     // x[] 
	mulss	20(%r11), %xmm0        // value
	addss	%xmm0, %xmm10           // xmm10 = sum
	cmpl	$6, %ecx              
	je	.L4                       // 6 left

  movslq 24(%rdx),%rax        // col_idx[jj]
	movss	(%r8,%rax,4), %xmm0     // x[] 
	mulss	24(%r11), %xmm0        // value
	addss	%xmm0, %xmm10           // xmm10 = sum
	cmpl	$7, %ecx              
	je	.L4                       // 7 left

.L4:
  vmulss (%rdi),%xmm15,%xmm1        // sum = y[i] * beta + 0 
	incl	%esi                        // Next row
  vmulss	%xmm10, %xmm14,%xmm13           // sum * alpha
	addss	%xmm13, %xmm1           
	movss	%xmm1, (%rdi)        // Write back y[i]
	jmp	.L17
.L10:
	popq	%rbx
	popq	%rbp
	popq	%r12
	popq	%r13
	popq	%r14
	ret
.LFE5279:
	.size	__spmv_csr_serial_host_avx2_float, .-__spmv_csr_serial_host_avx2_float
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
