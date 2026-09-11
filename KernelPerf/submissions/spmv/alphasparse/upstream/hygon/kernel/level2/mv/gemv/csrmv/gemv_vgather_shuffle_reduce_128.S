	.text
  .globl __spmv_csr_serial_host_gather4_float
	.type	__spmv_csr_serial_host_gather4_float, @function
// regester used :
// register unused: %r15
// void __spmv_csr_serial_host_gather4_float(
//                             const float alpha,   // xmm0
//                             const float beta,    // xmm1
//                             const int num_rows,         //rdi
//                             const int * Ap, //row_start //rsi
//                             const int * Aj, //col_idx   //rdx 
//                             const float * Ax, //value   //rcx
//                             const float * x,    //r8
//                                   float * y)    //r9
__spmv_csr_serial_host_gather4_float:
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
	movups	.MASK(%rip), %xmm5  // Eight 0xffffffff

  movaps %xmm0, %xmm14 // alpha
  movaps %xmm1, %xmm15 // beta

.L17:

// reg : c expression

/* loop var :
* esi: outer loop variable, row index i
 * ebp : rows
 * rdx : i * 4

* r10d: inner loop variable, nz index ai
 * ecx : rows_end[i] - 3
 */

/* addr
 * 
 * r12 : y
 * rdi : y + i
 * r8: x
 * rbx : rows_start
 * r14 : col_indx
* rdx: current col_idx (col_idx + ai)
 * r13 : values
* r11: current values (values + ai)
*/
	cmpl	%ebp, %esi  // Same as sub %ebp, %esi (esi - ebp)
	jge	.L10
	movslq	%esi,%rax            // i: row index
	leaq	0(,%rax,4), %rdx       //i*4
	movl	(%rbx,%rax,4), %r10d   // row_start[i]
	movl	4(%rbx,%rdx), %ecx     // row_start[i+1]

	leaq	(%r12,%rdx), %rdi      // y[i] addr
  xorps %xmm8,%xmm8
  xorps %xmm10,%xmm10

  subl  $3, %ecx                  // unroll four, ecx = rows_end[i] - 3
	movslq	%r10d,%rax              // row_start[i]
	salq	$2, %rax                  // addr of row_start

	leaq	(%rax,%r14), %rdx         // col_idx  addr
	leaq	(%rax,%r13), %r11         // value addr
	cmpl	%ecx, %r10d               // 
	jge	.L7
	.p2align 4,,7

// ########################inner loop########################
.L6:
 // x -> xmm2
 // values -> xmm6

  movdqu (%rdx),%xmm0          // col_indx[0,1,2,3]
  addl   $4,  %r10d           // r10d is the loop variable, incremented by 4 each time
  movups  (%r11), %xmm6       // value
  addq   $16, %rdx            // rdx is &col_idx[i], +16 each time
  movaps %xmm5,%xmm3          // mask bit
  VGATHERDPS %xmm3,(%r8,%xmm0,4),%xmm2 // xmm3: mask, r8: base address of x, xmm0: col_idx, xmm2: destination register
	vfmadd231ps	%xmm6, %xmm2, %xmm8 // xmm8 is the intermediate accumulator,
  addq  $16, %r11       // r11 is &value[i], +16 each time
	cmpl	%r10d, %ecx     // ecx = nnz_row - 3
	jg	.L6
// ########################end loop########################
  // Lines 121-128 reduce xmm8, with the result in xmm10
#if 1
  movaps    %xmm8, %xmm9                                 
  shufps    $0xb1,%xmm8,%xmm9
  addps     %xmm8,%xmm9
  movaps    %xmm9,%xmm10
  unpckhps  %xmm9,%xmm10
  addss     %xmm9,%xmm10
#endif
.L7:
  subl %r10d, %ecx           // On exit, ecx + 3 - r10d is the number of remaining nnz, possible values 0,1,2,3; that is, ecx - r10d may be -3, -2, -1, 0
  cmpl	$-3, %ecx           
	jle	.L4                    // 0 left

	movslq	(%rdx),%rax        // col_idx[jj]
	movss	(%r8,%rax,4), %xmm0  // x[] 
  vfmadd231ss (%r11), %xmm0, %xmm10 // val[0] * x[0] + accu

  cmpl	$-2, %ecx           
	jle	.L4                      // 1 left

  movslq	0x4(%rdx),%r9        // col_idx[jj]
	movss	(%r8,%r9,4), %xmm2     // x[] 
  vfmadd231ss 0x4(%r11), %xmm2, %xmm10 // val[1] * x[1] + accu
	cmpl	$-1, %ecx               
	jle	.L4                      // 2 left

  movslq	0x8(%rdx),%rcx       // col_idx[jj]
	movss	(%r8,%rcx,4), %xmm3    // x[] 
  vfmadd231ss 0x8(%r11), %xmm3, %xmm10 // val[2] * x[2] + accu

.L4:
  vmulss (%rdi),%xmm15,%xmm1        // y[i] * beta
	incl	%esi                        // Next row
  vfmadd231ss %xmm10, %xmm14, %xmm1 // y[i] = sum * alpha + y[i] 
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
	.size	__spmv_csr_serial_host_gather4_float, .-__spmv_csr_serial_host_gather4_float
	.section	.rodata.cst32,"aM",@progbits,32
	.align 32
.MASK:
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
