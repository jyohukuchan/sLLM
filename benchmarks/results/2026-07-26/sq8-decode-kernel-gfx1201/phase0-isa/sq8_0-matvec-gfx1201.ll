; ModuleID = '/home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-26/sq8-decode-kernel-gfx1201/phase0-isa/sq8_0_matvec_hiprtc_static.hip.cpp'
source_filename = "/home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-26/sq8-decode-kernel-gfx1201/phase0-isa/sq8_0_matvec_hiprtc_static.hip.cpp"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

@_ZZ29ullm_sq_fp8_matvec_f32_kernelE7partial = internal unnamed_addr addrspace(3) global [256 x float] undef, align 16
@_ZZ35ullm_sq_fp8_matvec_batch_f32_kernelE7partial = internal unnamed_addr addrspace(3) global [256 x float] undef, align 16
@_ZZ34ullm_sq_fp8_matvec_pair_f32_kernelE7partial = internal unnamed_addr addrspace(3) global [256 x float] undef, align 16
@_ZZ36ullm_sq_fp8_matvec_triple_f32_kernelE7partial = internal unnamed_addr addrspace(3) global [256 x float] undef, align 16
@__hip_cuid_46137f5c4cd8da57 = addrspace(1) global i8 0
@llvm.compiler.used = appending addrspace(1) global [1 x ptr] [ptr addrspacecast (ptr addrspace(1) @__hip_cuid_46137f5c4cd8da57 to ptr)], section "llvm.metadata"

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare float @llvm.amdgcn.cvt.f32.fp8(i32, i32 immarg) #0

; Function Attrs: convergent mustprogress nofree norecurse nounwind
define protected amdgpu_kernel void @ullm_sq_fp8_matvec_f32_kernel(ptr addrspace(1) noundef readonly captures(none) %0, ptr addrspace(1) noundef readonly captures(none) %1, ptr addrspace(1) noundef readonly captures(none) %2, i64 noundef %3, i64 noundef %4, i32 noundef %5, i64 noundef %6, i64 noundef %7, ptr addrspace(1) noundef writeonly captures(none) %8) local_unnamed_addr #1 {
  %10 = tail call i32 @llvm.amdgcn.workgroup.id.x()
  %11 = tail call noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x()
  %12 = zext i32 %10 to i64
  %13 = icmp ule i64 %3, %12
  br i1 %13, label %57, label %14

14:                                               ; preds = %9
  %15 = mul i64 %4, %12
  %16 = icmp eq i32 %5, 2
  br i1 %16, label %17, label %21

17:                                               ; preds = %14
  %18 = add i64 %4, -1
  %19 = udiv i64 %18, %7
  %20 = add i64 %19, 1
  br label %21

21:                                               ; preds = %17, %14
  %22 = phi i64 [ %20, %17 ], [ 1, %14 ]
  %23 = zext nneg i32 %11 to i64
  %24 = icmp ugt i64 %4, %23
  br i1 %24, label %25, label %57

25:                                               ; preds = %21
  %26 = icmp eq i32 %5, 1
  %27 = getelementptr i8, ptr addrspace(1) %0, i64 %15
  %28 = tail call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %29 = getelementptr inbounds nuw i8, ptr addrspace(4) %28, i64 12
  %30 = load i16, ptr addrspace(4) %29, align 4, !tbaa !6
  %31 = zext i16 %30 to i64
  %32 = select i1 %26, i64 %12, i64 0
  %33 = getelementptr inbounds nuw float, ptr addrspace(1) %1, i64 %32
  br label %34

34:                                               ; preds = %25, %43
  %35 = phi float [ 0.000000e+00, %25 ], [ %54, %43 ]
  %36 = phi i64 [ %23, %25 ], [ %55, %43 ]
  br i1 %16, label %37, label %43

37:                                               ; preds = %34
  %38 = udiv i64 %12, %6
  %39 = mul i64 %38, %22
  %40 = udiv i64 %36, %7
  %41 = getelementptr float, ptr addrspace(1) %1, i64 %39
  %42 = getelementptr float, ptr addrspace(1) %41, i64 %40
  br label %43

43:                                               ; preds = %34, %37
  %44 = phi ptr addrspace(1) [ %42, %37 ], [ %33, %34 ]
  %45 = load float, ptr addrspace(1) %44, align 4, !tbaa !10
  %46 = getelementptr i8, ptr addrspace(1) %27, i64 %36
  %47 = load i8, ptr addrspace(1) %46, align 1, !tbaa !14
  %48 = zext i8 %47 to i32
  %49 = tail call contract noundef float @llvm.amdgcn.cvt.f32.fp8(i32 %48, i32 0)
  %50 = fmul contract float %45, %49
  %51 = getelementptr inbounds float, ptr addrspace(1) %2, i64 %36
  %52 = load float, ptr addrspace(1) %51, align 4, !tbaa !10
  %53 = fmul contract float %50, %52
  %54 = fadd contract float %35, %53
  %55 = add i64 %36, %31
  %56 = icmp ult i64 %55, %4
  br i1 %56, label %34, label %57, !llvm.loop !15

57:                                               ; preds = %43, %21, %9
  %58 = phi float [ 0.000000e+00, %9 ], [ 0.000000e+00, %21 ], [ %54, %43 ]
  %59 = getelementptr inbounds nuw float, ptr addrspace(3) @_ZZ29ullm_sq_fp8_matvec_f32_kernelE7partial, i32 %11
  store float %58, ptr addrspace(3) %59, align 4, !tbaa !10
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %60 = tail call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %61 = getelementptr inbounds nuw i8, ptr addrspace(4) %60, i64 12
  %62 = load i16, ptr addrspace(4) %61, align 4, !tbaa !6
  %63 = icmp ult i16 %62, 2
  br i1 %63, label %67, label %64

64:                                               ; preds = %57
  %65 = lshr i16 %62, 1
  %66 = zext nneg i16 %65 to i32
  br label %70

67:                                               ; preds = %78, %57
  %68 = icmp ne i32 %11, 0
  %69 = or i1 %68, %13
  br i1 %69, label %84, label %81

70:                                               ; preds = %64, %78
  %71 = phi i32 [ %66, %64 ], [ %79, %78 ]
  %72 = icmp samesign ult i32 %11, %71
  br i1 %72, label %73, label %78

73:                                               ; preds = %70
  %74 = getelementptr inbounds nuw float, ptr addrspace(3) %59, i32 %71
  %75 = load float, ptr addrspace(3) %74, align 4, !tbaa !10
  %76 = load float, ptr addrspace(3) %59, align 4, !tbaa !10
  %77 = fadd contract float %75, %76
  store float %77, ptr addrspace(3) %59, align 4, !tbaa !10
  br label %78

78:                                               ; preds = %73, %70
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %79 = lshr i32 %71, 1
  %80 = icmp samesign ult i32 %71, 2
  br i1 %80, label %67, label %70, !llvm.loop !17

81:                                               ; preds = %67
  %82 = getelementptr inbounds nuw float, ptr addrspace(1) %8, i64 %12
  %83 = load float, ptr addrspace(3) @_ZZ29ullm_sq_fp8_matvec_f32_kernelE7partial, align 16, !tbaa !10
  store float %83, ptr addrspace(1) %82, align 4, !tbaa !10
  br label %84

84:                                               ; preds = %67, %81
  ret void
}

; Function Attrs: convergent mustprogress nofree norecurse nounwind
define protected amdgpu_kernel void @ullm_sq_fp8_matvec_batch_f32_kernel(ptr addrspace(1) noundef readonly captures(none) %0, ptr addrspace(1) noundef readonly captures(none) %1, ptr addrspace(1) noundef readonly captures(none) %2, i64 noundef %3, i64 noundef %4, i32 noundef %5, i64 noundef %6, i64 noundef %7, i64 noundef %8, ptr addrspace(1) noundef writeonly captures(none) %9) local_unnamed_addr #2 {
  %11 = tail call i32 @llvm.amdgcn.workgroup.id.x()
  %12 = tail call noundef i32 @llvm.amdgcn.workgroup.id.y()
  %13 = tail call noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x()
  %14 = zext i32 %11 to i64
  %15 = icmp ule i64 %3, %14
  br i1 %15, label %64, label %16

16:                                               ; preds = %10
  %17 = zext i32 %12 to i64
  %18 = icmp ugt i64 %8, %17
  br i1 %18, label %19, label %64

19:                                               ; preds = %16
  %20 = mul i64 %4, %14
  %21 = mul i64 %4, %17
  %22 = icmp eq i32 %5, 2
  br i1 %22, label %23, label %27

23:                                               ; preds = %19
  %24 = add i64 %4, -1
  %25 = udiv i64 %24, %7
  %26 = add i64 %25, 1
  br label %27

27:                                               ; preds = %23, %19
  %28 = phi i64 [ %26, %23 ], [ 1, %19 ]
  %29 = zext nneg i32 %13 to i64
  %30 = icmp ugt i64 %4, %29
  br i1 %30, label %31, label %64

31:                                               ; preds = %27
  %32 = icmp eq i32 %5, 1
  %33 = getelementptr i8, ptr addrspace(1) %0, i64 %20
  %34 = getelementptr float, ptr addrspace(1) %2, i64 %21
  %35 = tail call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %36 = getelementptr inbounds nuw i8, ptr addrspace(4) %35, i64 12
  %37 = load i16, ptr addrspace(4) %36, align 4, !tbaa !6
  %38 = zext i16 %37 to i64
  %39 = select i1 %32, i64 %14, i64 0
  %40 = getelementptr inbounds nuw float, ptr addrspace(1) %1, i64 %39
  br label %41

41:                                               ; preds = %31, %50
  %42 = phi float [ 0.000000e+00, %31 ], [ %61, %50 ]
  %43 = phi i64 [ %29, %31 ], [ %62, %50 ]
  br i1 %22, label %44, label %50

44:                                               ; preds = %41
  %45 = udiv i64 %14, %6
  %46 = mul i64 %45, %28
  %47 = udiv i64 %43, %7
  %48 = getelementptr float, ptr addrspace(1) %1, i64 %46
  %49 = getelementptr float, ptr addrspace(1) %48, i64 %47
  br label %50

50:                                               ; preds = %41, %44
  %51 = phi ptr addrspace(1) [ %49, %44 ], [ %40, %41 ]
  %52 = load float, ptr addrspace(1) %51, align 4, !tbaa !10
  %53 = getelementptr i8, ptr addrspace(1) %33, i64 %43
  %54 = load i8, ptr addrspace(1) %53, align 1, !tbaa !14
  %55 = zext i8 %54 to i32
  %56 = tail call contract noundef float @llvm.amdgcn.cvt.f32.fp8(i32 %55, i32 0)
  %57 = fmul contract float %52, %56
  %58 = getelementptr float, ptr addrspace(1) %34, i64 %43
  %59 = load float, ptr addrspace(1) %58, align 4, !tbaa !10
  %60 = fmul contract float %57, %59
  %61 = fadd contract float %42, %60
  %62 = add i64 %43, %38
  %63 = icmp ult i64 %62, %4
  br i1 %63, label %41, label %64, !llvm.loop !18

64:                                               ; preds = %50, %27, %16, %10
  %65 = phi float [ 0.000000e+00, %16 ], [ 0.000000e+00, %10 ], [ 0.000000e+00, %27 ], [ %61, %50 ]
  %66 = getelementptr inbounds nuw float, ptr addrspace(3) @_ZZ35ullm_sq_fp8_matvec_batch_f32_kernelE7partial, i32 %13
  store float %65, ptr addrspace(3) %66, align 4, !tbaa !10
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %67 = tail call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %68 = getelementptr inbounds nuw i8, ptr addrspace(4) %67, i64 12
  %69 = load i16, ptr addrspace(4) %68, align 4, !tbaa !6
  %70 = icmp ult i16 %69, 2
  br i1 %70, label %74, label %71

71:                                               ; preds = %64
  %72 = lshr i16 %69, 1
  %73 = zext nneg i16 %72 to i32
  br label %77

74:                                               ; preds = %85, %64
  %75 = icmp ne i32 %13, 0
  %76 = or i1 %75, %15
  br i1 %76, label %96, label %88

77:                                               ; preds = %71, %85
  %78 = phi i32 [ %73, %71 ], [ %86, %85 ]
  %79 = icmp samesign ult i32 %13, %78
  br i1 %79, label %80, label %85

80:                                               ; preds = %77
  %81 = getelementptr inbounds nuw float, ptr addrspace(3) %66, i32 %78
  %82 = load float, ptr addrspace(3) %81, align 4, !tbaa !10
  %83 = load float, ptr addrspace(3) %66, align 4, !tbaa !10
  %84 = fadd contract float %82, %83
  store float %84, ptr addrspace(3) %66, align 4, !tbaa !10
  br label %85

85:                                               ; preds = %80, %77
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %86 = lshr i32 %78, 1
  %87 = icmp samesign ult i32 %78, 2
  br i1 %87, label %74, label %77, !llvm.loop !19

88:                                               ; preds = %74
  %89 = zext i32 %12 to i64
  %90 = icmp ugt i64 %8, %89
  br i1 %90, label %91, label %96

91:                                               ; preds = %88
  %92 = load float, ptr addrspace(3) @_ZZ35ullm_sq_fp8_matvec_batch_f32_kernelE7partial, align 16, !tbaa !10
  %93 = mul i64 %3, %89
  %94 = getelementptr float, ptr addrspace(1) %9, i64 %93
  %95 = getelementptr float, ptr addrspace(1) %94, i64 %14
  store float %92, ptr addrspace(1) %95, align 4, !tbaa !10
  br label %96

96:                                               ; preds = %74, %91, %88
  ret void
}

; Function Attrs: convergent mustprogress nofree norecurse nounwind
define protected amdgpu_kernel void @ullm_sq_fp8_matvec_pair_f32_kernel(ptr addrspace(1) noundef readonly captures(none) %0, ptr addrspace(1) noundef readonly captures(none) %1, i64 noundef %2, i32 noundef %3, i64 noundef %4, ptr addrspace(1) noundef readonly captures(none) %5, ptr addrspace(1) noundef readonly captures(none) %6, i64 noundef %7, i32 noundef %8, i64 noundef %9, ptr addrspace(1) noundef readonly captures(none) %10, i64 noundef %11, ptr addrspace(1) noundef writeonly captures(none) %12, ptr addrspace(1) noundef writeonly captures(none) %13) local_unnamed_addr #2 {
  %15 = tail call i32 @llvm.amdgcn.workgroup.id.x()
  %16 = tail call noundef i32 @llvm.amdgcn.workgroup.id.y()
  %17 = tail call noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x()
  %18 = icmp eq i32 %16, 0
  %19 = select i1 %18, ptr addrspace(1) %0, ptr addrspace(1) %5
  %20 = select i1 %18, ptr addrspace(1) %1, ptr addrspace(1) %6
  %21 = select i1 %18, i64 %2, i64 %7
  %22 = select i1 %18, i32 %3, i32 %8
  %23 = select i1 %18, i64 %4, i64 %9
  %24 = zext i32 %15 to i64
  %25 = icmp ule i64 %21, %24
  br i1 %25, label %68, label %26

26:                                               ; preds = %14
  %27 = mul i64 %11, %24
  %28 = icmp eq i32 %22, 2
  br i1 %28, label %29, label %33

29:                                               ; preds = %26
  %30 = add i64 %23, -1
  %31 = add i64 %30, %11
  %32 = udiv i64 %31, %23
  br label %33

33:                                               ; preds = %29, %26
  %34 = phi i64 [ %32, %29 ], [ 1, %26 ]
  %35 = zext nneg i32 %17 to i64
  %36 = icmp ugt i64 %11, %35
  br i1 %36, label %37, label %68

37:                                               ; preds = %33
  %38 = icmp eq i32 %22, 1
  %39 = mul i64 %34, %24
  %40 = getelementptr float, ptr addrspace(1) %20, i64 %39
  %41 = getelementptr i8, ptr addrspace(1) %19, i64 %27
  %42 = tail call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %43 = getelementptr inbounds nuw i8, ptr addrspace(4) %42, i64 12
  %44 = load i16, ptr addrspace(4) %43, align 4, !tbaa !6
  %45 = zext i16 %44 to i64
  %46 = select i1 %38, i64 %24, i64 0
  %47 = getelementptr inbounds nuw float, ptr addrspace(1) %20, i64 %46
  br label %48

48:                                               ; preds = %37, %54
  %49 = phi float [ 0.000000e+00, %37 ], [ %65, %54 ]
  %50 = phi i64 [ %35, %37 ], [ %66, %54 ]
  br i1 %28, label %51, label %54

51:                                               ; preds = %48
  %52 = udiv i64 %50, %23
  %53 = getelementptr float, ptr addrspace(1) %40, i64 %52
  br label %54

54:                                               ; preds = %48, %51
  %55 = phi ptr addrspace(1) [ %53, %51 ], [ %47, %48 ]
  %56 = load float, ptr addrspace(1) %55, align 4, !tbaa !10
  %57 = getelementptr i8, ptr addrspace(1) %41, i64 %50
  %58 = load i8, ptr addrspace(1) %57, align 1, !tbaa !14
  %59 = zext i8 %58 to i32
  %60 = tail call contract noundef float @llvm.amdgcn.cvt.f32.fp8(i32 %59, i32 0)
  %61 = fmul contract float %56, %60
  %62 = getelementptr inbounds float, ptr addrspace(1) %10, i64 %50
  %63 = load float, ptr addrspace(1) %62, align 4, !tbaa !10
  %64 = fmul contract float %61, %63
  %65 = fadd contract float %49, %64
  %66 = add i64 %50, %45
  %67 = icmp ult i64 %66, %11
  br i1 %67, label %48, label %68, !llvm.loop !20

68:                                               ; preds = %54, %33, %14
  %69 = phi float [ 0.000000e+00, %14 ], [ 0.000000e+00, %33 ], [ %65, %54 ]
  %70 = getelementptr inbounds nuw float, ptr addrspace(3) @_ZZ34ullm_sq_fp8_matvec_pair_f32_kernelE7partial, i32 %17
  store float %69, ptr addrspace(3) %70, align 4, !tbaa !10
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %71 = tail call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %72 = getelementptr inbounds nuw i8, ptr addrspace(4) %71, i64 12
  %73 = load i16, ptr addrspace(4) %72, align 4, !tbaa !6
  %74 = icmp ult i16 %73, 2
  br i1 %74, label %78, label %75

75:                                               ; preds = %68
  %76 = lshr i16 %73, 1
  %77 = zext nneg i16 %76 to i32
  br label %81

78:                                               ; preds = %89, %68
  %79 = icmp ne i32 %17, 0
  %80 = or i1 %79, %25
  br i1 %80, label %96, label %92

81:                                               ; preds = %75, %89
  %82 = phi i32 [ %77, %75 ], [ %90, %89 ]
  %83 = icmp samesign ult i32 %17, %82
  br i1 %83, label %84, label %89

84:                                               ; preds = %81
  %85 = getelementptr inbounds nuw float, ptr addrspace(3) %70, i32 %82
  %86 = load float, ptr addrspace(3) %85, align 4, !tbaa !10
  %87 = load float, ptr addrspace(3) %70, align 4, !tbaa !10
  %88 = fadd contract float %86, %87
  store float %88, ptr addrspace(3) %70, align 4, !tbaa !10
  br label %89

89:                                               ; preds = %84, %81
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %90 = lshr i32 %82, 1
  %91 = icmp samesign ult i32 %82, 2
  br i1 %91, label %78, label %81, !llvm.loop !21

92:                                               ; preds = %78
  %93 = load float, ptr addrspace(3) @_ZZ34ullm_sq_fp8_matvec_pair_f32_kernelE7partial, align 16, !tbaa !10
  %94 = select i1 %18, ptr addrspace(1) %12, ptr addrspace(1) %13
  %95 = getelementptr inbounds nuw float, ptr addrspace(1) %94, i64 %24
  store float %93, ptr addrspace(1) %95, align 4, !tbaa !10
  br label %96

96:                                               ; preds = %92, %78
  ret void
}

; Function Attrs: convergent mustprogress nofree norecurse nounwind
define protected amdgpu_kernel void @ullm_sq_fp8_matvec_triple_f32_kernel(ptr addrspace(1) noundef readonly captures(none) %0, ptr addrspace(1) noundef readonly captures(none) %1, i64 noundef %2, i32 noundef %3, i64 noundef %4, ptr addrspace(1) noundef readonly captures(none) %5, ptr addrspace(1) noundef readonly captures(none) %6, i64 noundef %7, i32 noundef %8, i64 noundef %9, ptr addrspace(1) noundef readonly captures(none) %10, ptr addrspace(1) noundef readonly captures(none) %11, i64 noundef %12, i32 noundef %13, i64 noundef %14, ptr addrspace(1) noundef readonly captures(none) %15, i64 noundef %16, ptr addrspace(1) noundef writeonly captures(none) %17, ptr addrspace(1) noundef writeonly captures(none) %18, ptr addrspace(1) noundef writeonly captures(none) %19) local_unnamed_addr #2 {
  %21 = tail call i32 @llvm.amdgcn.workgroup.id.x()
  %22 = tail call noundef i32 @llvm.amdgcn.workgroup.id.y()
  %23 = tail call noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x()
  %24 = icmp eq i32 %22, 0
  %25 = icmp eq i32 %22, 1
  %26 = select i1 %25, ptr addrspace(1) %5, ptr addrspace(1) %10
  %27 = select i1 %24, ptr addrspace(1) %0, ptr addrspace(1) %26
  %28 = select i1 %25, ptr addrspace(1) %6, ptr addrspace(1) %11
  %29 = select i1 %24, ptr addrspace(1) %1, ptr addrspace(1) %28
  %30 = select i1 %25, i64 %7, i64 %12
  %31 = select i1 %24, i64 %2, i64 %30
  %32 = select i1 %25, i32 %8, i32 %13
  %33 = select i1 %24, i32 %3, i32 %32
  %34 = select i1 %25, i64 %9, i64 %14
  %35 = select i1 %24, i64 %4, i64 %34
  %36 = zext i32 %21 to i64
  %37 = icmp ule i64 %31, %36
  br i1 %37, label %80, label %38

38:                                               ; preds = %20
  %39 = mul i64 %16, %36
  %40 = icmp eq i32 %33, 2
  br i1 %40, label %41, label %45

41:                                               ; preds = %38
  %42 = add i64 %35, -1
  %43 = add i64 %42, %16
  %44 = udiv i64 %43, %35
  br label %45

45:                                               ; preds = %41, %38
  %46 = phi i64 [ %44, %41 ], [ 1, %38 ]
  %47 = zext nneg i32 %23 to i64
  %48 = icmp ugt i64 %16, %47
  br i1 %48, label %49, label %80

49:                                               ; preds = %45
  %50 = icmp eq i32 %33, 1
  %51 = mul i64 %46, %36
  %52 = getelementptr float, ptr addrspace(1) %29, i64 %51
  %53 = getelementptr i8, ptr addrspace(1) %27, i64 %39
  %54 = tail call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %55 = getelementptr inbounds nuw i8, ptr addrspace(4) %54, i64 12
  %56 = load i16, ptr addrspace(4) %55, align 4, !tbaa !6
  %57 = zext i16 %56 to i64
  %58 = select i1 %50, i64 %36, i64 0
  %59 = getelementptr inbounds nuw float, ptr addrspace(1) %29, i64 %58
  br label %60

60:                                               ; preds = %49, %66
  %61 = phi float [ 0.000000e+00, %49 ], [ %77, %66 ]
  %62 = phi i64 [ %47, %49 ], [ %78, %66 ]
  br i1 %40, label %63, label %66

63:                                               ; preds = %60
  %64 = udiv i64 %62, %35
  %65 = getelementptr float, ptr addrspace(1) %52, i64 %64
  br label %66

66:                                               ; preds = %60, %63
  %67 = phi ptr addrspace(1) [ %65, %63 ], [ %59, %60 ]
  %68 = load float, ptr addrspace(1) %67, align 4, !tbaa !10
  %69 = getelementptr i8, ptr addrspace(1) %53, i64 %62
  %70 = load i8, ptr addrspace(1) %69, align 1, !tbaa !14
  %71 = zext i8 %70 to i32
  %72 = tail call contract noundef float @llvm.amdgcn.cvt.f32.fp8(i32 %71, i32 0)
  %73 = fmul contract float %68, %72
  %74 = getelementptr inbounds float, ptr addrspace(1) %15, i64 %62
  %75 = load float, ptr addrspace(1) %74, align 4, !tbaa !10
  %76 = fmul contract float %73, %75
  %77 = fadd contract float %61, %76
  %78 = add i64 %62, %57
  %79 = icmp ult i64 %78, %16
  br i1 %79, label %60, label %80, !llvm.loop !22

80:                                               ; preds = %66, %45, %20
  %81 = phi float [ 0.000000e+00, %20 ], [ 0.000000e+00, %45 ], [ %77, %66 ]
  %82 = getelementptr inbounds nuw float, ptr addrspace(3) @_ZZ36ullm_sq_fp8_matvec_triple_f32_kernelE7partial, i32 %23
  store float %81, ptr addrspace(3) %82, align 4, !tbaa !10
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %83 = tail call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %84 = getelementptr inbounds nuw i8, ptr addrspace(4) %83, i64 12
  %85 = load i16, ptr addrspace(4) %84, align 4, !tbaa !6
  %86 = icmp ult i16 %85, 2
  br i1 %86, label %90, label %87

87:                                               ; preds = %80
  %88 = lshr i16 %85, 1
  %89 = zext nneg i16 %88 to i32
  br label %93

90:                                               ; preds = %101, %80
  %91 = icmp ne i32 %23, 0
  %92 = or i1 %91, %37
  br i1 %92, label %109, label %104

93:                                               ; preds = %87, %101
  %94 = phi i32 [ %89, %87 ], [ %102, %101 ]
  %95 = icmp samesign ult i32 %23, %94
  br i1 %95, label %96, label %101

96:                                               ; preds = %93
  %97 = getelementptr inbounds nuw float, ptr addrspace(3) %82, i32 %94
  %98 = load float, ptr addrspace(3) %97, align 4, !tbaa !10
  %99 = load float, ptr addrspace(3) %82, align 4, !tbaa !10
  %100 = fadd contract float %98, %99
  store float %100, ptr addrspace(3) %82, align 4, !tbaa !10
  br label %101

101:                                              ; preds = %96, %93
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %102 = lshr i32 %94, 1
  %103 = icmp samesign ult i32 %94, 2
  br i1 %103, label %90, label %93, !llvm.loop !23

104:                                              ; preds = %90
  %105 = load float, ptr addrspace(3) @_ZZ36ullm_sq_fp8_matvec_triple_f32_kernelE7partial, align 16, !tbaa !10
  %106 = select i1 %25, ptr addrspace(1) %18, ptr addrspace(1) %19
  %107 = select i1 %24, ptr addrspace(1) %17, ptr addrspace(1) %106
  %108 = getelementptr inbounds nuw float, ptr addrspace(1) %107, i64 %36
  store float %105, ptr addrspace(1) %108, align 4, !tbaa !10
  br label %109

109:                                              ; preds = %104, %90
  ret void
}

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef align 4 ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr() #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #4

attributes #0 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
attributes #1 = { convergent mustprogress nofree norecurse nounwind "amdgpu-agpr-alloc"="0" "amdgpu-flat-work-group-size"="1,1024" "amdgpu-no-cluster-id-x" "amdgpu-no-cluster-id-y" "amdgpu-no-cluster-id-z" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1201" "target-features"="+16-bit-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-buffer-pk-add-bf16-inst,+atomic-ds-pk-add-16-insts,+atomic-fadd-rtn-insts,+atomic-flat-pk-add-16-insts,+atomic-fmin-fmax-global-f32,+atomic-global-pk-add-bf16-inst,+ci-insts,+dl-insts,+dot10-insts,+dot11-insts,+dot12-insts,+dot7-insts,+dot8-insts,+dot9-insts,+dpp,+fp8-conversion-insts,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx12-insts,+gfx8-insts,+gfx9-insts,+wavefrontsize32" "uniform-work-group-size"="true" }
attributes #2 = { convergent mustprogress nofree norecurse nounwind "amdgpu-agpr-alloc"="0" "amdgpu-flat-work-group-size"="1,1024" "amdgpu-no-cluster-id-x" "amdgpu-no-cluster-id-y" "amdgpu-no-cluster-id-z" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1201" "target-features"="+16-bit-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-buffer-pk-add-bf16-inst,+atomic-ds-pk-add-16-insts,+atomic-fadd-rtn-insts,+atomic-flat-pk-add-16-insts,+atomic-fmin-fmax-global-f32,+atomic-global-pk-add-bf16-inst,+ci-insts,+dl-insts,+dot10-insts,+dot11-insts,+dot12-insts,+dot7-insts,+dot8-insts,+dot9-insts,+dpp,+fp8-conversion-insts,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx12-insts,+gfx8-insts,+gfx9-insts,+wavefrontsize32" "uniform-work-group-size"="true" }
attributes #3 = { convergent mustprogress nocallback nofree nounwind willreturn }
attributes #4 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}
!opencl.ocl.version = !{!5}

!0 = !{i32 1, !"amdhsa_code_object_version", i32 600}
!1 = !{i32 1, !"amdgpu_printf_kind", !"hostcall"}
!2 = !{i32 1, !"wchar_size", i32 4}
!3 = !{i32 8, !"PIC Level", i32 2}
!4 = !{!"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.1 26084 f58b06dce1f9c15707c5f808fd002e18c2accf7e)"}
!5 = !{i32 2, i32 0}
!6 = !{!7, !7, i64 0}
!7 = !{!"short", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!11, !11, i64 0}
!11 = !{!"float", !12, i64 0}
!12 = !{!"omnipotent char", !13, i64 0}
!13 = !{!"Simple C++ TBAA"}
!14 = !{!12, !12, i64 0}
!15 = distinct !{!15, !16}
!16 = !{!"llvm.loop.mustprogress"}
!17 = distinct !{!17, !16}
!18 = distinct !{!18, !16}
!19 = distinct !{!19, !16}
!20 = distinct !{!20, !16}
!21 = distinct !{!21, !16}
!22 = distinct !{!22, !16}
!23 = distinct !{!23, !16}
