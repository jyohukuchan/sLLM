; ModuleID = '/home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-26/sq8-decode-kernel-gfx1201/phase0-isa/sq8_0_matvec_hiprtc_static.hip.cpp'
source_filename = "/home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-26/sq8-decode-kernel-gfx1201/phase0-isa/sq8_0_matvec_hiprtc_static.hip.cpp"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

@_ZZ29ullm_sq_fp8_matvec_f32_kernelE12wave_partial = internal unnamed_addr addrspace(3) global [8 x float] undef, align 16
@_ZZ35ullm_sq_fp8_matvec_batch_f32_kernelE12wave_partial = internal unnamed_addr addrspace(3) global [8 x float] undef, align 16
@_ZZ34ullm_sq_fp8_matvec_pair_f32_kernelE7partial = internal unnamed_addr addrspace(3) global [256 x float] undef, align 16
@_ZZ36ullm_sq_fp8_matvec_triple_f32_kernelE7partial = internal unnamed_addr addrspace(3) global [256 x float] undef, align 16
@__hip_cuid_dd36914507823da4 = addrspace(1) global i8 0
@llvm.compiler.used = appending addrspace(1) global [1 x ptr] [ptr addrspacecast (ptr addrspace(1) @__hip_cuid_dd36914507823da4 to ptr)], section "llvm.metadata"

; Function Attrs: convergent mustprogress nofree norecurse nounwind
define protected amdgpu_kernel void @ullm_sq_fp8_matvec_f32_kernel(ptr addrspace(1) noundef %0, ptr addrspace(1) noundef readonly captures(none) %1, ptr addrspace(1) noundef readonly captures(none) %2, i64 noundef %3, i64 noundef %4, i32 noundef %5, i64 noundef %6, i64 noundef %7, ptr addrspace(1) noundef writeonly captures(none) %8) local_unnamed_addr #0 {
  %10 = tail call noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x()
  %11 = and i32 %10, 31
  %12 = lshr i32 %10, 5
  %13 = tail call i32 @llvm.amdgcn.workgroup.id.x()
  %14 = zext i32 %13 to i64
  %15 = icmp ule i64 %3, %14
  br i1 %15, label %182, label %16

16:                                               ; preds = %9
  %17 = mul i64 %4, %14
  %18 = getelementptr inbounds i8, ptr addrspace(1) %0, i64 %17
  %19 = addrspacecast ptr addrspace(1) %18 to ptr
  %20 = ptrtoint ptr %19 to i64
  %21 = icmp eq i32 %5, 2
  br i1 %21, label %22, label %26

22:                                               ; preds = %16
  %23 = add i64 %4, -1
  %24 = udiv i64 %23, %7
  %25 = add i64 %24, 1
  br label %26

26:                                               ; preds = %16, %22
  %27 = phi i64 [ %25, %22 ], [ 1, %16 ]
  br i1 %21, label %28, label %35

28:                                               ; preds = %26
  %29 = icmp ne i64 %6, 0
  %30 = icmp ugt i64 %7, 15
  %31 = and i1 %29, %30
  %32 = and i64 %7, 15
  %33 = icmp eq i64 %32, 0
  %34 = and i1 %31, %33
  br label %35

35:                                               ; preds = %28, %26
  %36 = phi i1 [ true, %26 ], [ %34, %28 ]
  %37 = add i64 %4, 15
  %38 = lshr i64 %37, 4
  %39 = zext nneg i32 %10 to i64
  %40 = icmp samesign ugt i64 %38, %39
  br i1 %40, label %41, label %182

41:                                               ; preds = %35
  %42 = getelementptr inbounds nuw float, ptr addrspace(1) %1, i64 %14
  %43 = tail call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %44 = getelementptr inbounds nuw i8, ptr addrspace(4) %43, i64 12
  %45 = load i16, ptr addrspace(4) %44, align 4, !tbaa !6
  %46 = zext i16 %45 to i64
  %47 = shl nuw nsw i64 %39, 4
  %48 = sub i64 %4, %47
  %49 = shl nuw nsw i64 %46, 4
  %50 = and i64 %20, 15
  %51 = icmp eq i64 %50, 0
  br label %52

52:                                               ; preds = %41, %177
  %53 = phi i64 [ %48, %41 ], [ %181, %177 ]
  %54 = phi float [ 0.000000e+00, %41 ], [ %178, %177 ]
  %55 = phi i64 [ %39, %41 ], [ %179, %177 ]
  %56 = tail call i64 @llvm.umin.i64(i64 %53, i64 16)
  %57 = trunc nuw nsw i64 %56 to i32
  %58 = tail call i32 @llvm.umax.i32(i32 %57, i32 1)
  %59 = shl nuw i64 %55, 4
  %60 = sub i64 %4, %59
  %61 = icmp ugt i64 %60, 15
  %62 = select i1 %36, i1 %61, i1 false
  %63 = select i1 %62, i1 %51, i1 false
  br i1 %63, label %66, label %64

64:                                               ; preds = %52
  %65 = icmp eq i64 %4, %59
  br i1 %65, label %177, label %128

66:                                               ; preds = %52
  switch i32 %5, label %74 [
    i32 1, label %67
    i32 2, label %68
  ]

67:                                               ; preds = %66
  br label %74

68:                                               ; preds = %66
  %69 = udiv i64 %14, %6
  %70 = mul i64 %69, %27
  %71 = udiv i64 %59, %7
  %72 = getelementptr float, ptr addrspace(1) %1, i64 %70
  %73 = getelementptr float, ptr addrspace(1) %72, i64 %71
  br label %74

74:                                               ; preds = %66, %67, %68
  %75 = phi ptr addrspace(1) [ %42, %67 ], [ %73, %68 ], [ %1, %66 ]
  %76 = load float, ptr addrspace(1) %75, align 4, !tbaa !10
  %77 = getelementptr inbounds i8, ptr addrspace(1) %18, i64 %59
  %78 = load <4 x i32>, ptr addrspace(1) %77, align 16, !tbaa !14
  %79 = getelementptr inbounds float, ptr addrspace(1) %2, i64 %59
  br label %80

80:                                               ; preds = %86, %74
  %81 = phi i32 [ 0, %74 ], [ %87, %86 ]
  %82 = phi float [ %54, %74 ], [ %124, %86 ]
  %83 = zext nneg i32 %81 to i64
  %84 = extractelement <4 x i32> %78, i64 %83
  %85 = shl nuw nsw i32 %81, 2
  br label %89

86:                                               ; preds = %117
  %87 = add nuw nsw i32 %81, 1
  %88 = icmp eq i32 %87, 4
  br i1 %88, label %177, label %80, !llvm.loop !15

89:                                               ; preds = %117, %80
  %90 = phi i32 [ 0, %80 ], [ %126, %117 ]
  %91 = phi i32 [ %84, %80 ], [ %125, %117 ]
  %92 = phi float [ %82, %80 ], [ %124, %117 ]
  %93 = lshr i32 %91, 3
  %94 = and i32 %93, 15
  %95 = and i32 %91, 7
  %96 = icmp eq i32 %94, 15
  %97 = icmp eq i32 %95, 7
  %98 = and i1 %97, %96
  br i1 %98, label %117, label %99

99:                                               ; preds = %89
  %100 = icmp eq i32 %94, 0
  br i1 %100, label %101, label %108

101:                                              ; preds = %99
  %102 = uitofp nneg i32 %95 to float
  %103 = fmul contract float %102, 0x3F60000000000000
  %104 = fneg contract float %103
  %105 = and i32 %91, 128
  %106 = icmp eq i32 %105, 0
  %107 = select contract i1 %106, float %103, float %104
  br label %117

108:                                              ; preds = %99
  %109 = shl i32 %91, 24
  %110 = and i32 %109, -2147483648
  %111 = shl nuw nsw i32 %94, 23
  %112 = add nuw nsw i32 %111, 1006632960
  %113 = or disjoint i32 %112, %110
  %114 = shl nuw nsw i32 %95, 20
  %115 = or disjoint i32 %113, %114
  %116 = bitcast i32 %115 to float
  br label %117

117:                                              ; preds = %108, %101, %89
  %118 = phi float [ %107, %101 ], [ %116, %108 ], [ 0x7FF8000000000000, %89 ]
  %119 = fmul contract float %76, %118
  %120 = add nuw nsw i32 %90, %85
  %121 = zext nneg i32 %120 to i64
  %122 = getelementptr inbounds nuw float, ptr addrspace(1) %79, i64 %121
  %123 = load float, ptr addrspace(1) %122, align 4, !tbaa !10
  %124 = tail call contract noundef float @llvm.fma.f32(float %119, float %123, float %92)
  %125 = lshr i32 %91, 8
  %126 = add nuw nsw i32 %90, 1
  %127 = icmp eq i32 %126, 4
  br i1 %127, label %86, label %89, !llvm.loop !18

128:                                              ; preds = %64, %169
  %129 = phi float [ %174, %169 ], [ %54, %64 ]
  %130 = phi i32 [ %175, %169 ], [ 0, %64 ]
  %131 = zext nneg i32 %130 to i64
  %132 = add i64 %59, %131
  switch i32 %5, label %140 [
    i32 1, label %133
    i32 2, label %134
  ]

133:                                              ; preds = %128
  br label %140

134:                                              ; preds = %128
  %135 = udiv i64 %14, %6
  %136 = mul i64 %135, %27
  %137 = udiv i64 %132, %7
  %138 = getelementptr float, ptr addrspace(1) %1, i64 %136
  %139 = getelementptr float, ptr addrspace(1) %138, i64 %137
  br label %140

140:                                              ; preds = %128, %133, %134
  %141 = phi ptr addrspace(1) [ %42, %133 ], [ %139, %134 ], [ %1, %128 ]
  %142 = load float, ptr addrspace(1) %141, align 4, !tbaa !10
  %143 = getelementptr inbounds i8, ptr addrspace(1) %18, i64 %132
  %144 = load i8, ptr addrspace(1) %143, align 1, !tbaa !14
  %145 = zext i8 %144 to i32
  %146 = lshr i32 %145, 3
  %147 = and i32 %146, 15
  %148 = and i32 %145, 7
  %149 = icmp eq i32 %147, 15
  %150 = icmp eq i32 %148, 7
  %151 = and i1 %150, %149
  br i1 %151, label %169, label %152

152:                                              ; preds = %140
  %153 = icmp eq i32 %147, 0
  br i1 %153, label %154, label %160

154:                                              ; preds = %152
  %155 = uitofp nneg i32 %148 to float
  %156 = fmul contract float %155, 0x3F60000000000000
  %157 = fneg contract float %156
  %158 = icmp slt i8 %144, 0
  %159 = select contract i1 %158, float %157, float %156
  br label %169

160:                                              ; preds = %152
  %161 = sext i8 %144 to i32
  %162 = and i32 %161, -2147483648
  %163 = shl nuw nsw i32 %147, 23
  %164 = add nuw nsw i32 %163, 1006632960
  %165 = or disjoint i32 %164, %162
  %166 = shl nuw nsw i32 %148, 20
  %167 = or disjoint i32 %165, %166
  %168 = bitcast i32 %167 to float
  br label %169

169:                                              ; preds = %140, %154, %160
  %170 = phi float [ %159, %154 ], [ %168, %160 ], [ 0x7FF8000000000000, %140 ]
  %171 = fmul contract float %142, %170
  %172 = getelementptr inbounds float, ptr addrspace(1) %2, i64 %132
  %173 = load float, ptr addrspace(1) %172, align 4, !tbaa !10
  %174 = tail call contract noundef float @llvm.fma.f32(float %171, float %173, float %129)
  %175 = add nuw nsw i32 %130, 1
  %176 = icmp eq i32 %175, %58
  br i1 %176, label %177, label %128, !llvm.loop !19

177:                                              ; preds = %169, %86, %64
  %178 = phi float [ %54, %64 ], [ %124, %86 ], [ %174, %169 ]
  %179 = add nuw nsw i64 %55, %46
  %180 = icmp samesign ult i64 %179, %38
  %181 = sub i64 %53, %49
  br i1 %180, label %52, label %182, !llvm.loop !20

182:                                              ; preds = %177, %35, %9
  %183 = phi float [ 0.000000e+00, %9 ], [ 0.000000e+00, %35 ], [ %178, %177 ]
  %184 = tail call noundef i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %185 = and i32 %184, 31
  %186 = xor i32 %185, 31
  %187 = bitcast float %183 to i32
  %188 = and i32 %186, 16
  %189 = add i32 %188, %184
  %190 = shl i32 %189, 2
  %191 = tail call noundef i32 @llvm.amdgcn.ds.bpermute(i32 %190, i32 %187)
  %192 = bitcast i32 %191 to float
  %193 = fadd contract float %183, %192
  %194 = bitcast float %193 to i32
  %195 = icmp samesign ult i32 %186, 8
  %196 = select i1 %195, i32 0, i32 8
  %197 = add i32 %196, %184
  %198 = shl i32 %197, 2
  %199 = tail call noundef i32 @llvm.amdgcn.ds.bpermute(i32 %198, i32 %194)
  %200 = bitcast i32 %199 to float
  %201 = fadd contract float %193, %200
  %202 = bitcast float %201 to i32
  %203 = icmp samesign ult i32 %186, 4
  %204 = select i1 %203, i32 0, i32 4
  %205 = add i32 %204, %184
  %206 = shl i32 %205, 2
  %207 = tail call noundef i32 @llvm.amdgcn.ds.bpermute(i32 %206, i32 %202)
  %208 = bitcast i32 %207 to float
  %209 = fadd contract float %201, %208
  %210 = bitcast float %209 to i32
  %211 = icmp samesign ult i32 %186, 2
  %212 = select i1 %211, i32 0, i32 2
  %213 = add i32 %212, %184
  %214 = shl i32 %213, 2
  %215 = tail call noundef i32 @llvm.amdgcn.ds.bpermute(i32 %214, i32 %210)
  %216 = bitcast i32 %215 to float
  %217 = fadd contract float %209, %216
  %218 = bitcast float %217 to i32
  %219 = icmp ne i32 %185, 31
  %220 = zext i1 %219 to i32
  %221 = add i32 %184, %220
  %222 = shl i32 %221, 2
  %223 = tail call noundef i32 @llvm.amdgcn.ds.bpermute(i32 %222, i32 %218)
  %224 = icmp eq i32 %11, 0
  br i1 %224, label %225, label %229

225:                                              ; preds = %182
  %226 = getelementptr inbounds nuw float, ptr addrspace(3) @_ZZ29ullm_sq_fp8_matvec_f32_kernelE12wave_partial, i32 %12
  %227 = bitcast i32 %223 to float
  %228 = fadd contract float %217, %227
  store float %228, ptr addrspace(3) %226, align 4, !tbaa !10
  br label %229

229:                                              ; preds = %225, %182
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %230 = icmp ne i32 %10, 0
  %231 = or i1 %230, %15
  br i1 %231, label %250, label %232

232:                                              ; preds = %229
  %233 = load <2 x float>, ptr addrspace(3) @_ZZ29ullm_sq_fp8_matvec_f32_kernelE12wave_partial, align 16, !tbaa !10
  %234 = extractelement <2 x float> %233, i64 0
  %235 = extractelement <2 x float> %233, i64 1
  %236 = fadd contract float %234, %235
  %237 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZ29ullm_sq_fp8_matvec_f32_kernelE12wave_partial, i32 8), align 8, !tbaa !10
  %238 = fadd contract float %236, %237
  %239 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZ29ullm_sq_fp8_matvec_f32_kernelE12wave_partial, i32 12), align 4, !tbaa !10
  %240 = fadd contract float %238, %239
  %241 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZ29ullm_sq_fp8_matvec_f32_kernelE12wave_partial, i32 16), align 16, !tbaa !10
  %242 = fadd contract float %240, %241
  %243 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZ29ullm_sq_fp8_matvec_f32_kernelE12wave_partial, i32 20), align 4, !tbaa !10
  %244 = fadd contract float %242, %243
  %245 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZ29ullm_sq_fp8_matvec_f32_kernelE12wave_partial, i32 24), align 8, !tbaa !10
  %246 = fadd contract float %244, %245
  %247 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZ29ullm_sq_fp8_matvec_f32_kernelE12wave_partial, i32 28), align 4, !tbaa !10
  %248 = fadd contract float %246, %247
  %249 = getelementptr inbounds nuw float, ptr addrspace(1) %8, i64 %14
  store float %248, ptr addrspace(1) %249, align 4, !tbaa !10
  br label %250

250:                                              ; preds = %229, %232
  ret void
}

; Function Attrs: convergent mustprogress nofree norecurse nounwind
define protected amdgpu_kernel void @ullm_sq_fp8_matvec_batch_f32_kernel(ptr addrspace(1) noundef %0, ptr addrspace(1) noundef readonly captures(none) %1, ptr addrspace(1) noundef readonly captures(none) %2, i64 noundef %3, i64 noundef %4, i32 noundef %5, i64 noundef %6, i64 noundef %7, i64 noundef %8, ptr addrspace(1) noundef writeonly captures(none) %9) local_unnamed_addr #1 {
  %11 = tail call noundef i32 @llvm.amdgcn.workgroup.id.y()
  %12 = zext i32 %11 to i64
  %13 = tail call noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x()
  %14 = and i32 %13, 31
  %15 = lshr i32 %13, 5
  %16 = tail call i32 @llvm.amdgcn.workgroup.id.x()
  %17 = zext i32 %16 to i64
  %18 = icmp ugt i64 %3, %17
  %19 = icmp ugt i64 %8, %12
  %20 = and i1 %18, %19
  br i1 %20, label %21, label %189

21:                                               ; preds = %10
  %22 = mul i64 %4, %17
  %23 = getelementptr inbounds i8, ptr addrspace(1) %0, i64 %22
  %24 = addrspacecast ptr addrspace(1) %23 to ptr
  %25 = ptrtoint ptr %24 to i64
  %26 = mul i64 %4, %12
  %27 = getelementptr inbounds float, ptr addrspace(1) %2, i64 %26
  %28 = icmp eq i32 %5, 2
  br i1 %28, label %29, label %33

29:                                               ; preds = %21
  %30 = add i64 %4, -1
  %31 = udiv i64 %30, %7
  %32 = add i64 %31, 1
  br label %33

33:                                               ; preds = %21, %29
  %34 = phi i64 [ %32, %29 ], [ 1, %21 ]
  br i1 %28, label %35, label %42

35:                                               ; preds = %33
  %36 = icmp ne i64 %6, 0
  %37 = icmp ugt i64 %7, 15
  %38 = and i1 %36, %37
  %39 = and i64 %7, 15
  %40 = icmp eq i64 %39, 0
  %41 = and i1 %38, %40
  br label %42

42:                                               ; preds = %35, %33
  %43 = phi i1 [ true, %33 ], [ %41, %35 ]
  %44 = add i64 %4, 15
  %45 = lshr i64 %44, 4
  %46 = zext nneg i32 %13 to i64
  %47 = icmp samesign ugt i64 %45, %46
  br i1 %47, label %48, label %189

48:                                               ; preds = %42
  %49 = getelementptr inbounds nuw float, ptr addrspace(1) %1, i64 %17
  %50 = tail call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %51 = getelementptr inbounds nuw i8, ptr addrspace(4) %50, i64 12
  %52 = load i16, ptr addrspace(4) %51, align 4, !tbaa !6
  %53 = zext i16 %52 to i64
  %54 = shl nuw nsw i64 %46, 4
  %55 = sub i64 %4, %54
  %56 = shl nuw nsw i64 %53, 4
  %57 = and i64 %25, 15
  %58 = icmp eq i64 %57, 0
  br label %59

59:                                               ; preds = %48, %184
  %60 = phi i64 [ %55, %48 ], [ %188, %184 ]
  %61 = phi float [ 0.000000e+00, %48 ], [ %185, %184 ]
  %62 = phi i64 [ %46, %48 ], [ %186, %184 ]
  %63 = tail call i64 @llvm.umin.i64(i64 %60, i64 16)
  %64 = trunc nuw nsw i64 %63 to i32
  %65 = tail call i32 @llvm.umax.i32(i32 %64, i32 1)
  %66 = shl nuw i64 %62, 4
  %67 = sub i64 %4, %66
  %68 = icmp ugt i64 %67, 15
  %69 = select i1 %43, i1 %68, i1 false
  %70 = select i1 %69, i1 %58, i1 false
  br i1 %70, label %73, label %71

71:                                               ; preds = %59
  %72 = icmp eq i64 %4, %66
  br i1 %72, label %184, label %135

73:                                               ; preds = %59
  switch i32 %5, label %81 [
    i32 1, label %74
    i32 2, label %75
  ]

74:                                               ; preds = %73
  br label %81

75:                                               ; preds = %73
  %76 = udiv i64 %17, %6
  %77 = mul i64 %76, %34
  %78 = udiv i64 %66, %7
  %79 = getelementptr float, ptr addrspace(1) %1, i64 %77
  %80 = getelementptr float, ptr addrspace(1) %79, i64 %78
  br label %81

81:                                               ; preds = %73, %74, %75
  %82 = phi ptr addrspace(1) [ %49, %74 ], [ %80, %75 ], [ %1, %73 ]
  %83 = load float, ptr addrspace(1) %82, align 4, !tbaa !10
  %84 = getelementptr inbounds i8, ptr addrspace(1) %23, i64 %66
  %85 = load <4 x i32>, ptr addrspace(1) %84, align 16, !tbaa !14
  %86 = getelementptr inbounds float, ptr addrspace(1) %27, i64 %66
  br label %87

87:                                               ; preds = %93, %81
  %88 = phi i32 [ 0, %81 ], [ %94, %93 ]
  %89 = phi float [ %61, %81 ], [ %131, %93 ]
  %90 = zext nneg i32 %88 to i64
  %91 = extractelement <4 x i32> %85, i64 %90
  %92 = shl nuw nsw i32 %88, 2
  br label %96

93:                                               ; preds = %124
  %94 = add nuw nsw i32 %88, 1
  %95 = icmp eq i32 %94, 4
  br i1 %95, label %184, label %87, !llvm.loop !15

96:                                               ; preds = %124, %87
  %97 = phi i32 [ 0, %87 ], [ %133, %124 ]
  %98 = phi i32 [ %91, %87 ], [ %132, %124 ]
  %99 = phi float [ %89, %87 ], [ %131, %124 ]
  %100 = lshr i32 %98, 3
  %101 = and i32 %100, 15
  %102 = and i32 %98, 7
  %103 = icmp eq i32 %101, 15
  %104 = icmp eq i32 %102, 7
  %105 = and i1 %104, %103
  br i1 %105, label %124, label %106

106:                                              ; preds = %96
  %107 = icmp eq i32 %101, 0
  br i1 %107, label %108, label %115

108:                                              ; preds = %106
  %109 = uitofp nneg i32 %102 to float
  %110 = fmul contract float %109, 0x3F60000000000000
  %111 = fneg contract float %110
  %112 = and i32 %98, 128
  %113 = icmp eq i32 %112, 0
  %114 = select contract i1 %113, float %110, float %111
  br label %124

115:                                              ; preds = %106
  %116 = shl i32 %98, 24
  %117 = and i32 %116, -2147483648
  %118 = shl nuw nsw i32 %101, 23
  %119 = add nuw nsw i32 %118, 1006632960
  %120 = or disjoint i32 %119, %117
  %121 = shl nuw nsw i32 %102, 20
  %122 = or disjoint i32 %120, %121
  %123 = bitcast i32 %122 to float
  br label %124

124:                                              ; preds = %115, %108, %96
  %125 = phi float [ %114, %108 ], [ %123, %115 ], [ 0x7FF8000000000000, %96 ]
  %126 = fmul contract float %83, %125
  %127 = add nuw nsw i32 %97, %92
  %128 = zext nneg i32 %127 to i64
  %129 = getelementptr inbounds nuw float, ptr addrspace(1) %86, i64 %128
  %130 = load float, ptr addrspace(1) %129, align 4, !tbaa !10
  %131 = tail call contract noundef float @llvm.fma.f32(float %126, float %130, float %99)
  %132 = lshr i32 %98, 8
  %133 = add nuw nsw i32 %97, 1
  %134 = icmp eq i32 %133, 4
  br i1 %134, label %93, label %96, !llvm.loop !18

135:                                              ; preds = %71, %176
  %136 = phi float [ %181, %176 ], [ %61, %71 ]
  %137 = phi i32 [ %182, %176 ], [ 0, %71 ]
  %138 = zext nneg i32 %137 to i64
  %139 = add i64 %66, %138
  switch i32 %5, label %147 [
    i32 1, label %140
    i32 2, label %141
  ]

140:                                              ; preds = %135
  br label %147

141:                                              ; preds = %135
  %142 = udiv i64 %17, %6
  %143 = mul i64 %142, %34
  %144 = udiv i64 %139, %7
  %145 = getelementptr float, ptr addrspace(1) %1, i64 %143
  %146 = getelementptr float, ptr addrspace(1) %145, i64 %144
  br label %147

147:                                              ; preds = %135, %140, %141
  %148 = phi ptr addrspace(1) [ %49, %140 ], [ %146, %141 ], [ %1, %135 ]
  %149 = load float, ptr addrspace(1) %148, align 4, !tbaa !10
  %150 = getelementptr inbounds i8, ptr addrspace(1) %23, i64 %139
  %151 = load i8, ptr addrspace(1) %150, align 1, !tbaa !14
  %152 = zext i8 %151 to i32
  %153 = lshr i32 %152, 3
  %154 = and i32 %153, 15
  %155 = and i32 %152, 7
  %156 = icmp eq i32 %154, 15
  %157 = icmp eq i32 %155, 7
  %158 = and i1 %157, %156
  br i1 %158, label %176, label %159

159:                                              ; preds = %147
  %160 = icmp eq i32 %154, 0
  br i1 %160, label %161, label %167

161:                                              ; preds = %159
  %162 = uitofp nneg i32 %155 to float
  %163 = fmul contract float %162, 0x3F60000000000000
  %164 = fneg contract float %163
  %165 = icmp slt i8 %151, 0
  %166 = select contract i1 %165, float %164, float %163
  br label %176

167:                                              ; preds = %159
  %168 = sext i8 %151 to i32
  %169 = and i32 %168, -2147483648
  %170 = shl nuw nsw i32 %154, 23
  %171 = add nuw nsw i32 %170, 1006632960
  %172 = or disjoint i32 %171, %169
  %173 = shl nuw nsw i32 %155, 20
  %174 = or disjoint i32 %172, %173
  %175 = bitcast i32 %174 to float
  br label %176

176:                                              ; preds = %147, %161, %167
  %177 = phi float [ %166, %161 ], [ %175, %167 ], [ 0x7FF8000000000000, %147 ]
  %178 = fmul contract float %149, %177
  %179 = getelementptr inbounds float, ptr addrspace(1) %27, i64 %139
  %180 = load float, ptr addrspace(1) %179, align 4, !tbaa !10
  %181 = tail call contract noundef float @llvm.fma.f32(float %178, float %180, float %136)
  %182 = add nuw nsw i32 %137, 1
  %183 = icmp eq i32 %182, %65
  br i1 %183, label %184, label %135, !llvm.loop !21

184:                                              ; preds = %176, %93, %71
  %185 = phi float [ %61, %71 ], [ %131, %93 ], [ %181, %176 ]
  %186 = add nuw nsw i64 %62, %53
  %187 = icmp samesign ult i64 %186, %45
  %188 = sub i64 %60, %56
  br i1 %187, label %59, label %189, !llvm.loop !22

189:                                              ; preds = %184, %42, %10
  %190 = phi float [ 0.000000e+00, %10 ], [ 0.000000e+00, %42 ], [ %185, %184 ]
  %191 = tail call noundef i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %192 = and i32 %191, 31
  %193 = xor i32 %192, 31
  %194 = bitcast float %190 to i32
  %195 = and i32 %193, 16
  %196 = add i32 %195, %191
  %197 = shl i32 %196, 2
  %198 = tail call noundef i32 @llvm.amdgcn.ds.bpermute(i32 %197, i32 %194)
  %199 = bitcast i32 %198 to float
  %200 = fadd contract float %190, %199
  %201 = bitcast float %200 to i32
  %202 = icmp samesign ult i32 %193, 8
  %203 = select i1 %202, i32 0, i32 8
  %204 = add i32 %203, %191
  %205 = shl i32 %204, 2
  %206 = tail call noundef i32 @llvm.amdgcn.ds.bpermute(i32 %205, i32 %201)
  %207 = bitcast i32 %206 to float
  %208 = fadd contract float %200, %207
  %209 = bitcast float %208 to i32
  %210 = icmp samesign ult i32 %193, 4
  %211 = select i1 %210, i32 0, i32 4
  %212 = add i32 %211, %191
  %213 = shl i32 %212, 2
  %214 = tail call noundef i32 @llvm.amdgcn.ds.bpermute(i32 %213, i32 %209)
  %215 = bitcast i32 %214 to float
  %216 = fadd contract float %208, %215
  %217 = bitcast float %216 to i32
  %218 = icmp samesign ult i32 %193, 2
  %219 = select i1 %218, i32 0, i32 2
  %220 = add i32 %219, %191
  %221 = shl i32 %220, 2
  %222 = tail call noundef i32 @llvm.amdgcn.ds.bpermute(i32 %221, i32 %217)
  %223 = bitcast i32 %222 to float
  %224 = fadd contract float %216, %223
  %225 = bitcast float %224 to i32
  %226 = icmp ne i32 %192, 31
  %227 = zext i1 %226 to i32
  %228 = add i32 %191, %227
  %229 = shl i32 %228, 2
  %230 = tail call noundef i32 @llvm.amdgcn.ds.bpermute(i32 %229, i32 %225)
  %231 = icmp eq i32 %14, 0
  br i1 %231, label %232, label %236

232:                                              ; preds = %189
  %233 = getelementptr inbounds nuw float, ptr addrspace(3) @_ZZ35ullm_sq_fp8_matvec_batch_f32_kernelE12wave_partial, i32 %15
  %234 = bitcast i32 %230 to float
  %235 = fadd contract float %224, %234
  store float %235, ptr addrspace(3) %233, align 4, !tbaa !10
  br label %236

236:                                              ; preds = %232, %189
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %237 = icmp eq i32 %13, 0
  %238 = and i1 %237, %18
  %239 = and i1 %238, %19
  br i1 %239, label %240, label %260

240:                                              ; preds = %236
  %241 = load <2 x float>, ptr addrspace(3) @_ZZ35ullm_sq_fp8_matvec_batch_f32_kernelE12wave_partial, align 16, !tbaa !10
  %242 = extractelement <2 x float> %241, i64 0
  %243 = extractelement <2 x float> %241, i64 1
  %244 = fadd contract float %242, %243
  %245 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZ35ullm_sq_fp8_matvec_batch_f32_kernelE12wave_partial, i32 8), align 8, !tbaa !10
  %246 = fadd contract float %244, %245
  %247 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZ35ullm_sq_fp8_matvec_batch_f32_kernelE12wave_partial, i32 12), align 4, !tbaa !10
  %248 = fadd contract float %246, %247
  %249 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZ35ullm_sq_fp8_matvec_batch_f32_kernelE12wave_partial, i32 16), align 16, !tbaa !10
  %250 = fadd contract float %248, %249
  %251 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZ35ullm_sq_fp8_matvec_batch_f32_kernelE12wave_partial, i32 20), align 4, !tbaa !10
  %252 = fadd contract float %250, %251
  %253 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZ35ullm_sq_fp8_matvec_batch_f32_kernelE12wave_partial, i32 24), align 8, !tbaa !10
  %254 = fadd contract float %252, %253
  %255 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZ35ullm_sq_fp8_matvec_batch_f32_kernelE12wave_partial, i32 28), align 4, !tbaa !10
  %256 = fadd contract float %254, %255
  %257 = mul i64 %3, %12
  %258 = getelementptr float, ptr addrspace(1) %9, i64 %257
  %259 = getelementptr float, ptr addrspace(1) %258, i64 %17
  store float %256, ptr addrspace(1) %259, align 4, !tbaa !10
  br label %260

260:                                              ; preds = %236, %240
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
  br i1 %25, label %92, label %26

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
  br i1 %36, label %37, label %92

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

48:                                               ; preds = %37, %83
  %49 = phi float [ 0.000000e+00, %37 ], [ %89, %83 ]
  %50 = phi i64 [ %35, %37 ], [ %90, %83 ]
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
  %60 = lshr i32 %59, 3
  %61 = and i32 %60, 15
  %62 = and i32 %59, 7
  %63 = icmp eq i32 %61, 15
  %64 = icmp eq i32 %62, 7
  %65 = and i1 %64, %63
  br i1 %65, label %83, label %66

66:                                               ; preds = %54
  %67 = icmp eq i32 %61, 0
  br i1 %67, label %68, label %74

68:                                               ; preds = %66
  %69 = uitofp nneg i32 %62 to float
  %70 = fmul contract float %69, 0x3F60000000000000
  %71 = fneg contract float %70
  %72 = icmp slt i8 %58, 0
  %73 = select contract i1 %72, float %71, float %70
  br label %83

74:                                               ; preds = %66
  %75 = sext i8 %58 to i32
  %76 = and i32 %75, -2147483648
  %77 = shl nuw nsw i32 %61, 23
  %78 = add nuw nsw i32 %77, 1006632960
  %79 = or disjoint i32 %78, %76
  %80 = shl nuw nsw i32 %62, 20
  %81 = or disjoint i32 %79, %80
  %82 = bitcast i32 %81 to float
  br label %83

83:                                               ; preds = %54, %68, %74
  %84 = phi float [ %73, %68 ], [ %82, %74 ], [ 0x7FF8000000000000, %54 ]
  %85 = fmul contract float %56, %84
  %86 = getelementptr inbounds float, ptr addrspace(1) %10, i64 %50
  %87 = load float, ptr addrspace(1) %86, align 4, !tbaa !10
  %88 = fmul contract float %85, %87
  %89 = fadd contract float %49, %88
  %90 = add i64 %50, %45
  %91 = icmp ult i64 %90, %11
  br i1 %91, label %48, label %92, !llvm.loop !23

92:                                               ; preds = %83, %33, %14
  %93 = phi float [ 0.000000e+00, %14 ], [ 0.000000e+00, %33 ], [ %89, %83 ]
  %94 = getelementptr inbounds nuw float, ptr addrspace(3) @_ZZ34ullm_sq_fp8_matvec_pair_f32_kernelE7partial, i32 %17
  store float %93, ptr addrspace(3) %94, align 4, !tbaa !10
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %95 = tail call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %96 = getelementptr inbounds nuw i8, ptr addrspace(4) %95, i64 12
  %97 = load i16, ptr addrspace(4) %96, align 4, !tbaa !6
  %98 = icmp ult i16 %97, 2
  br i1 %98, label %102, label %99

99:                                               ; preds = %92
  %100 = lshr i16 %97, 1
  %101 = zext nneg i16 %100 to i32
  br label %105

102:                                              ; preds = %113, %92
  %103 = icmp ne i32 %17, 0
  %104 = or i1 %103, %25
  br i1 %104, label %120, label %116

105:                                              ; preds = %99, %113
  %106 = phi i32 [ %101, %99 ], [ %114, %113 ]
  %107 = icmp samesign ult i32 %17, %106
  br i1 %107, label %108, label %113

108:                                              ; preds = %105
  %109 = getelementptr inbounds nuw float, ptr addrspace(3) %94, i32 %106
  %110 = load float, ptr addrspace(3) %109, align 4, !tbaa !10
  %111 = load float, ptr addrspace(3) %94, align 4, !tbaa !10
  %112 = fadd contract float %110, %111
  store float %112, ptr addrspace(3) %94, align 4, !tbaa !10
  br label %113

113:                                              ; preds = %108, %105
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %114 = lshr i32 %106, 1
  %115 = icmp samesign ult i32 %106, 2
  br i1 %115, label %102, label %105, !llvm.loop !24

116:                                              ; preds = %102
  %117 = load float, ptr addrspace(3) @_ZZ34ullm_sq_fp8_matvec_pair_f32_kernelE7partial, align 16, !tbaa !10
  %118 = select i1 %18, ptr addrspace(1) %12, ptr addrspace(1) %13
  %119 = getelementptr inbounds nuw float, ptr addrspace(1) %118, i64 %24
  store float %117, ptr addrspace(1) %119, align 4, !tbaa !10
  br label %120

120:                                              ; preds = %116, %102
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
  br i1 %37, label %104, label %38

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
  br i1 %48, label %49, label %104

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

60:                                               ; preds = %49, %95
  %61 = phi float [ 0.000000e+00, %49 ], [ %101, %95 ]
  %62 = phi i64 [ %47, %49 ], [ %102, %95 ]
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
  %72 = lshr i32 %71, 3
  %73 = and i32 %72, 15
  %74 = and i32 %71, 7
  %75 = icmp eq i32 %73, 15
  %76 = icmp eq i32 %74, 7
  %77 = and i1 %76, %75
  br i1 %77, label %95, label %78

78:                                               ; preds = %66
  %79 = icmp eq i32 %73, 0
  br i1 %79, label %80, label %86

80:                                               ; preds = %78
  %81 = uitofp nneg i32 %74 to float
  %82 = fmul contract float %81, 0x3F60000000000000
  %83 = fneg contract float %82
  %84 = icmp slt i8 %70, 0
  %85 = select contract i1 %84, float %83, float %82
  br label %95

86:                                               ; preds = %78
  %87 = sext i8 %70 to i32
  %88 = and i32 %87, -2147483648
  %89 = shl nuw nsw i32 %73, 23
  %90 = add nuw nsw i32 %89, 1006632960
  %91 = or disjoint i32 %90, %88
  %92 = shl nuw nsw i32 %74, 20
  %93 = or disjoint i32 %91, %92
  %94 = bitcast i32 %93 to float
  br label %95

95:                                               ; preds = %66, %80, %86
  %96 = phi float [ %85, %80 ], [ %94, %86 ], [ 0x7FF8000000000000, %66 ]
  %97 = fmul contract float %68, %96
  %98 = getelementptr inbounds float, ptr addrspace(1) %15, i64 %62
  %99 = load float, ptr addrspace(1) %98, align 4, !tbaa !10
  %100 = fmul contract float %97, %99
  %101 = fadd contract float %61, %100
  %102 = add i64 %62, %57
  %103 = icmp ult i64 %102, %16
  br i1 %103, label %60, label %104, !llvm.loop !25

104:                                              ; preds = %95, %45, %20
  %105 = phi float [ 0.000000e+00, %20 ], [ 0.000000e+00, %45 ], [ %101, %95 ]
  %106 = getelementptr inbounds nuw float, ptr addrspace(3) @_ZZ36ullm_sq_fp8_matvec_triple_f32_kernelE7partial, i32 %23
  store float %105, ptr addrspace(3) %106, align 4, !tbaa !10
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %107 = tail call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %108 = getelementptr inbounds nuw i8, ptr addrspace(4) %107, i64 12
  %109 = load i16, ptr addrspace(4) %108, align 4, !tbaa !6
  %110 = icmp ult i16 %109, 2
  br i1 %110, label %114, label %111

111:                                              ; preds = %104
  %112 = lshr i16 %109, 1
  %113 = zext nneg i16 %112 to i32
  br label %117

114:                                              ; preds = %125, %104
  %115 = icmp ne i32 %23, 0
  %116 = or i1 %115, %37
  br i1 %116, label %133, label %128

117:                                              ; preds = %111, %125
  %118 = phi i32 [ %113, %111 ], [ %126, %125 ]
  %119 = icmp samesign ult i32 %23, %118
  br i1 %119, label %120, label %125

120:                                              ; preds = %117
  %121 = getelementptr inbounds nuw float, ptr addrspace(3) %106, i32 %118
  %122 = load float, ptr addrspace(3) %121, align 4, !tbaa !10
  %123 = load float, ptr addrspace(3) %106, align 4, !tbaa !10
  %124 = fadd contract float %122, %123
  store float %124, ptr addrspace(3) %106, align 4, !tbaa !10
  br label %125

125:                                              ; preds = %120, %117
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %126 = lshr i32 %118, 1
  %127 = icmp samesign ult i32 %118, 2
  br i1 %127, label %114, label %117, !llvm.loop !26

128:                                              ; preds = %114
  %129 = load float, ptr addrspace(3) @_ZZ36ullm_sq_fp8_matvec_triple_f32_kernelE7partial, align 16, !tbaa !10
  %130 = select i1 %25, ptr addrspace(1) %18, ptr addrspace(1) %19
  %131 = select i1 %24, ptr addrspace(1) %17, ptr addrspace(1) %130
  %132 = getelementptr inbounds nuw float, ptr addrspace(1) %131, i64 %36
  store float %129, ptr addrspace(1) %132, align 4, !tbaa !10
  br label %133

133:                                              ; preds = %128, %114
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fma.f32(float, float, float) #3

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #5

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #6

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef align 4 ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr() #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umin.i64(i64, i64) #7

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.umax.i32(i32, i32) #7

attributes #0 = { convergent mustprogress nofree norecurse nounwind "amdgpu-agpr-alloc"="0" "amdgpu-flat-work-group-size"="1,256" "amdgpu-no-cluster-id-x" "amdgpu-no-cluster-id-y" "amdgpu-no-cluster-id-z" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1030" "target-features"="+16-bit-insts,+atomic-fmin-fmax-global-f32,+atomic-fmin-fmax-global-f64,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize32" "uniform-work-group-size"="true" }
attributes #1 = { convergent mustprogress nofree norecurse nounwind "amdgpu-agpr-alloc"="0" "amdgpu-flat-work-group-size"="1,256" "amdgpu-no-cluster-id-x" "amdgpu-no-cluster-id-y" "amdgpu-no-cluster-id-z" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1030" "target-features"="+16-bit-insts,+atomic-fmin-fmax-global-f32,+atomic-fmin-fmax-global-f64,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize32" "uniform-work-group-size"="true" }
attributes #2 = { convergent mustprogress nofree norecurse nounwind "amdgpu-agpr-alloc"="0" "amdgpu-flat-work-group-size"="1,1024" "amdgpu-no-cluster-id-x" "amdgpu-no-cluster-id-y" "amdgpu-no-cluster-id-z" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1030" "target-features"="+16-bit-insts,+atomic-fmin-fmax-global-f32,+atomic-fmin-fmax-global-f64,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize32" "uniform-work-group-size"="true" }
attributes #3 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { convergent mustprogress nocallback nofree nounwind willreturn memory(none) }
attributes #5 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
attributes #6 = { convergent mustprogress nocallback nofree nounwind willreturn }
attributes #7 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

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
!15 = distinct !{!15, !16, !17}
!16 = !{!"llvm.loop.mustprogress"}
!17 = !{!"llvm.loop.unroll.disable"}
!18 = distinct !{!18, !16, !17}
!19 = distinct !{!19, !16}
!20 = distinct !{!20, !16}
!21 = distinct !{!21, !16}
!22 = distinct !{!22, !16}
!23 = distinct !{!23, !16}
!24 = distinct !{!24, !16}
!25 = distinct !{!25, !16}
!26 = distinct !{!26, !16}
