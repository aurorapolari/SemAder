; ModuleID = 'bin.c'
source_filename = "bin.c"
target datalayout = "e-m:x-p:32:32-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-w64-windows-gnu"

%union._LARGE_INTEGER = type { i64 }

@__const.main.arr = private unnamed_addr constant [50 x i32] [i32 3, i32 7, i32 1, i32 4, i32 2, i32 8, i32 5, i32 0, i32 9, i32 6, i32 3, i32 7, i32 1, i32 4, i32 2, i32 8, i32 5, i32 0, i32 9, i32 6, i32 3, i32 7, i32 1, i32 4, i32 2, i32 8, i32 5, i32 0, i32 9, i32 6, i32 3, i32 7, i32 1, i32 4, i32 2, i32 8, i32 5, i32 0, i32 9, i32 6, i32 3, i32 7, i32 1, i32 4, i32 2, i32 8, i32 5, i32 0, i32 9, i32 6], align 4
@.str = private unnamed_addr constant [4 x i8] c"%d \00", align 1
@.str.1 = private unnamed_addr constant [2 x i8] c"\0A\00", align 1
@.str.2 = private unnamed_addr constant [35 x i8] c"Time measured: %.3f milliseconds.\0A\00", align 1

; Function Attrs: noinline nounwind optnone
define dso_local void @swap(ptr noundef %0, ptr noundef %1) #0 {
  %3 = alloca ptr, align 4
  %4 = alloca ptr, align 4
  %5 = alloca i32, align 4
  store ptr %0, ptr %3, align 4
  store ptr %1, ptr %4, align 4
  %6 = load ptr, ptr %3, align 4
  %7 = load i32, ptr %6, align 4
  store i32 %7, ptr %5, align 4
  %8 = load ptr, ptr %4, align 4
  %9 = load i32, ptr %8, align 4
  %10 = load ptr, ptr %3, align 4
  store i32 %9, ptr %10, align 4
  %11 = load i32, ptr %5, align 4
  %12 = load ptr, ptr %4, align 4
  store i32 %11, ptr %12, align 4
  ret void
}

; Function Attrs: noinline nounwind optnone
define dso_local void @adjust_down(ptr noundef %0, i32 noundef %1, i32 noundef %2) #0 {
  %4 = alloca ptr, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  %9 = alloca i32, align 4
  %10 = alloca i32, align 4
  store ptr %0, ptr %4, align 4
  store i32 %1, ptr %5, align 4
  store i32 %2, ptr %6, align 4
  %11 = load i32, ptr %5, align 4
  store i32 %11, ptr %7, align 4
  %12 = load i32, ptr %5, align 4
  %13 = mul nsw i32 %12, 2
  %14 = add nsw i32 %13, 1
  store i32 %14, ptr %8, align 4
  %15 = load i32, ptr %8, align 4
  %16 = add nsw i32 %15, 1
  store i32 %16, ptr %9, align 4
  br label %17

17:                                               ; preds = %62, %3
  %18 = load i32, ptr %8, align 4
  %19 = load i32, ptr %6, align 4
  %20 = icmp slt i32 %18, %19
  br i1 %20, label %21, label %63

21:                                               ; preds = %17
  %22 = load i32, ptr %8, align 4
  store i32 %22, ptr %10, align 4
  %23 = load i32, ptr %9, align 4
  %24 = load i32, ptr %6, align 4
  %25 = icmp slt i32 %23, %24
  br i1 %25, label %26, label %38

26:                                               ; preds = %21
  %27 = load ptr, ptr %4, align 4
  %28 = load i32, ptr %9, align 4
  %29 = getelementptr inbounds i32, ptr %27, i32 %28
  %30 = load i32, ptr %29, align 4
  %31 = load ptr, ptr %4, align 4
  %32 = load i32, ptr %10, align 4
  %33 = getelementptr inbounds i32, ptr %31, i32 %32
  %34 = load i32, ptr %33, align 4
  %35 = icmp sgt i32 %30, %34
  br i1 %35, label %36, label %38

36:                                               ; preds = %26
  %37 = load i32, ptr %9, align 4
  store i32 %37, ptr %10, align 4
  br label %38

38:                                               ; preds = %36, %26, %21
  %39 = load ptr, ptr %4, align 4
  %40 = load i32, ptr %10, align 4
  %41 = getelementptr inbounds i32, ptr %39, i32 %40
  %42 = load i32, ptr %41, align 4
  %43 = load ptr, ptr %4, align 4
  %44 = load i32, ptr %7, align 4
  %45 = getelementptr inbounds i32, ptr %43, i32 %44
  %46 = load i32, ptr %45, align 4
  %47 = icmp sgt i32 %42, %46
  br i1 %47, label %48, label %61

48:                                               ; preds = %38
  %49 = load ptr, ptr %4, align 4
  %50 = load i32, ptr %10, align 4
  %51 = getelementptr inbounds i32, ptr %49, i32 %50
  %52 = load ptr, ptr %4, align 4
  %53 = load i32, ptr %7, align 4
  %54 = getelementptr inbounds i32, ptr %52, i32 %53
  call void @swap(ptr noundef %51, ptr noundef %54)
  %55 = load i32, ptr %10, align 4
  store i32 %55, ptr %7, align 4
  %56 = load i32, ptr %7, align 4
  %57 = mul nsw i32 %56, 2
  %58 = add nsw i32 %57, 1
  store i32 %58, ptr %8, align 4
  %59 = load i32, ptr %8, align 4
  %60 = add nsw i32 %59, 1
  store i32 %60, ptr %9, align 4
  br label %62

61:                                               ; preds = %38
  br label %63

62:                                               ; preds = %48
  br label %17, !llvm.loop !5

63:                                               ; preds = %61, %17
  ret void
}

; Function Attrs: noinline nounwind optnone
define dso_local void @heap_sort(ptr noundef %0, i32 noundef %1) #0 {
  %3 = alloca ptr, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  %9 = alloca i32, align 4
  %10 = alloca i32, align 4
  %11 = alloca i32, align 4
  %12 = alloca i32, align 4
  %13 = alloca i32, align 4
  %14 = alloca i32, align 4
  %15 = alloca i32, align 4
  %16 = alloca i32, align 4
  store ptr %0, ptr %3, align 4
  store i32 %1, ptr %4, align 4
  store i32 0, ptr %5, align 4
  %17 = load i32, ptr %4, align 4
  %18 = sdiv i32 %17, 2
  %19 = sub nsw i32 %18, 1
  store i32 %19, ptr %5, align 4
  br label %20

20:                                               ; preds = %86, %2
  %21 = load i32, ptr %5, align 4
  %22 = icmp sge i32 %21, 0
  br i1 %22, label %23, label %89

23:                                               ; preds = %20
  %24 = load i32, ptr %5, align 4
  store i32 %24, ptr %6, align 4
  %25 = load i32, ptr %5, align 4
  %26 = mul nsw i32 %25, 2
  %27 = add nsw i32 %26, 1
  store i32 %27, ptr %7, align 4
  %28 = load i32, ptr %7, align 4
  %29 = add nsw i32 %28, 1
  store i32 %29, ptr %8, align 4
  br label %30

30:                                               ; preds = %84, %23
  %31 = load i32, ptr %7, align 4
  %32 = load i32, ptr %4, align 4
  %33 = icmp slt i32 %31, %32
  br i1 %33, label %34, label %85

34:                                               ; preds = %30
  %35 = load i32, ptr %7, align 4
  store i32 %35, ptr %9, align 4
  %36 = load i32, ptr %8, align 4
  %37 = load i32, ptr %4, align 4
  %38 = icmp slt i32 %36, %37
  br i1 %38, label %39, label %51

39:                                               ; preds = %34
  %40 = load ptr, ptr %3, align 4
  %41 = load i32, ptr %8, align 4
  %42 = getelementptr inbounds i32, ptr %40, i32 %41
  %43 = load i32, ptr %42, align 4
  %44 = load ptr, ptr %3, align 4
  %45 = load i32, ptr %9, align 4
  %46 = getelementptr inbounds i32, ptr %44, i32 %45
  %47 = load i32, ptr %46, align 4
  %48 = icmp sgt i32 %43, %47
  br i1 %48, label %49, label %51

49:                                               ; preds = %39
  %50 = load i32, ptr %8, align 4
  store i32 %50, ptr %9, align 4
  br label %51

51:                                               ; preds = %49, %39, %34
  %52 = load ptr, ptr %3, align 4
  %53 = load i32, ptr %9, align 4
  %54 = getelementptr inbounds i32, ptr %52, i32 %53
  %55 = load i32, ptr %54, align 4
  %56 = load ptr, ptr %3, align 4
  %57 = load i32, ptr %6, align 4
  %58 = getelementptr inbounds i32, ptr %56, i32 %57
  %59 = load i32, ptr %58, align 4
  %60 = icmp sgt i32 %55, %59
  br i1 %60, label %61, label %83

61:                                               ; preds = %51
  %62 = load ptr, ptr %3, align 4
  %63 = load i32, ptr %9, align 4
  %64 = getelementptr inbounds i32, ptr %62, i32 %63
  %65 = load i32, ptr %64, align 4
  store i32 %65, ptr %10, align 4
  %66 = load ptr, ptr %3, align 4
  %67 = load i32, ptr %6, align 4
  %68 = getelementptr inbounds i32, ptr %66, i32 %67
  %69 = load i32, ptr %68, align 4
  %70 = load ptr, ptr %3, align 4
  %71 = load i32, ptr %9, align 4
  %72 = getelementptr inbounds i32, ptr %70, i32 %71
  store i32 %69, ptr %72, align 4
  %73 = load i32, ptr %10, align 4
  %74 = load ptr, ptr %3, align 4
  %75 = load i32, ptr %6, align 4
  %76 = getelementptr inbounds i32, ptr %74, i32 %75
  store i32 %73, ptr %76, align 4
  %77 = load i32, ptr %9, align 4
  store i32 %77, ptr %6, align 4
  %78 = load i32, ptr %6, align 4
  %79 = mul nsw i32 %78, 2
  %80 = add nsw i32 %79, 1
  store i32 %80, ptr %7, align 4
  %81 = load i32, ptr %7, align 4
  %82 = add nsw i32 %81, 1
  store i32 %82, ptr %8, align 4
  br label %84

83:                                               ; preds = %51
  br label %85

84:                                               ; preds = %61
  br label %30, !llvm.loop !7

85:                                               ; preds = %83, %30
  br label %86

86:                                               ; preds = %85
  %87 = load i32, ptr %5, align 4
  %88 = add nsw i32 %87, -1
  store i32 %88, ptr %5, align 4
  br label %20, !llvm.loop !8

89:                                               ; preds = %20
  %90 = load i32, ptr %4, align 4
  %91 = sub nsw i32 %90, 1
  store i32 %91, ptr %11, align 4
  br label %92

92:                                               ; preds = %166, %89
  %93 = load i32, ptr %11, align 4
  %94 = icmp sgt i32 %93, 0
  br i1 %94, label %95, label %169

95:                                               ; preds = %92
  %96 = load ptr, ptr %3, align 4
  %97 = getelementptr inbounds i32, ptr %96, i32 0
  %98 = load i32, ptr %97, align 4
  store i32 %98, ptr %12, align 4
  %99 = load ptr, ptr %3, align 4
  %100 = load i32, ptr %11, align 4
  %101 = getelementptr inbounds i32, ptr %99, i32 %100
  %102 = load i32, ptr %101, align 4
  %103 = load ptr, ptr %3, align 4
  %104 = getelementptr inbounds i32, ptr %103, i32 0
  store i32 %102, ptr %104, align 4
  %105 = load i32, ptr %12, align 4
  %106 = load ptr, ptr %3, align 4
  %107 = load i32, ptr %11, align 4
  %108 = getelementptr inbounds i32, ptr %106, i32 %107
  store i32 %105, ptr %108, align 4
  store i32 0, ptr %13, align 4
  store i32 1, ptr %14, align 4
  %109 = load i32, ptr %14, align 4
  %110 = add nsw i32 %109, 1
  store i32 %110, ptr %15, align 4
  br label %111

111:                                              ; preds = %165, %95
  %112 = load i32, ptr %14, align 4
  %113 = load i32, ptr %11, align 4
  %114 = icmp slt i32 %112, %113
  br i1 %114, label %115, label %166

115:                                              ; preds = %111
  %116 = load i32, ptr %14, align 4
  store i32 %116, ptr %16, align 4
  %117 = load i32, ptr %15, align 4
  %118 = load i32, ptr %11, align 4
  %119 = icmp slt i32 %117, %118
  br i1 %119, label %120, label %132

120:                                              ; preds = %115
  %121 = load ptr, ptr %3, align 4
  %122 = load i32, ptr %15, align 4
  %123 = getelementptr inbounds i32, ptr %121, i32 %122
  %124 = load i32, ptr %123, align 4
  %125 = load ptr, ptr %3, align 4
  %126 = load i32, ptr %16, align 4
  %127 = getelementptr inbounds i32, ptr %125, i32 %126
  %128 = load i32, ptr %127, align 4
  %129 = icmp sgt i32 %124, %128
  br i1 %129, label %130, label %132

130:                                              ; preds = %120
  %131 = load i32, ptr %15, align 4
  store i32 %131, ptr %16, align 4
  br label %132

132:                                              ; preds = %130, %120, %115
  %133 = load ptr, ptr %3, align 4
  %134 = load i32, ptr %16, align 4
  %135 = getelementptr inbounds i32, ptr %133, i32 %134
  %136 = load i32, ptr %135, align 4
  %137 = load ptr, ptr %3, align 4
  %138 = load i32, ptr %13, align 4
  %139 = getelementptr inbounds i32, ptr %137, i32 %138
  %140 = load i32, ptr %139, align 4
  %141 = icmp sgt i32 %136, %140
  br i1 %141, label %142, label %164

142:                                              ; preds = %132
  %143 = load ptr, ptr %3, align 4
  %144 = load i32, ptr %16, align 4
  %145 = getelementptr inbounds i32, ptr %143, i32 %144
  %146 = load i32, ptr %145, align 4
  store i32 %146, ptr %12, align 4
  %147 = load ptr, ptr %3, align 4
  %148 = load i32, ptr %13, align 4
  %149 = getelementptr inbounds i32, ptr %147, i32 %148
  %150 = load i32, ptr %149, align 4
  %151 = load ptr, ptr %3, align 4
  %152 = load i32, ptr %16, align 4
  %153 = getelementptr inbounds i32, ptr %151, i32 %152
  store i32 %150, ptr %153, align 4
  %154 = load i32, ptr %12, align 4
  %155 = load ptr, ptr %3, align 4
  %156 = load i32, ptr %13, align 4
  %157 = getelementptr inbounds i32, ptr %155, i32 %156
  store i32 %154, ptr %157, align 4
  %158 = load i32, ptr %16, align 4
  store i32 %158, ptr %13, align 4
  %159 = load i32, ptr %13, align 4
  %160 = mul nsw i32 %159, 2
  %161 = add nsw i32 %160, 1
  store i32 %161, ptr %14, align 4
  %162 = load i32, ptr %14, align 4
  %163 = add nsw i32 %162, 1
  store i32 %163, ptr %15, align 4
  br label %165

164:                                              ; preds = %132
  br label %166

165:                                              ; preds = %142
  br label %111, !llvm.loop !9

166:                                              ; preds = %164, %111
  %167 = load i32, ptr %11, align 4
  %168 = add nsw i32 %167, -1
  store i32 %168, ptr %11, align 4
  br label %92, !llvm.loop !10

169:                                              ; preds = %92
  ret void
}

; Function Attrs: noinline nounwind optnone
define dso_local i32 @main() #0 {
  %1 = alloca i32, align 4
  %2 = alloca %union._LARGE_INTEGER, align 8
  %3 = alloca %union._LARGE_INTEGER, align 8
  %4 = alloca %union._LARGE_INTEGER, align 8
  %5 = alloca double, align 8
  %6 = alloca [50 x i32], align 4
  %7 = alloca i32, align 4
  store i32 0, ptr %1, align 4
  %8 = call x86_stdcallcc i32 @"\01_QueryPerformanceFrequency@4"(ptr noundef %2)
  %9 = call x86_stdcallcc i32 @"\01_QueryPerformanceCounter@4"(ptr noundef %3)
  call void @llvm.memcpy.p0.p0.i32(ptr align 4 %6, ptr align 4 @__const.main.arr, i32 200, i1 false)
  %10 = getelementptr inbounds [50 x i32], ptr %6, i32 0, i32 0
  call void @heap_sort(ptr noundef %10, i32 noundef 50)
  store i32 0, ptr %7, align 4
  br label %11

11:                                               ; preds = %19, %0
  %12 = load i32, ptr %7, align 4
  %13 = icmp slt i32 %12, 50
  br i1 %13, label %14, label %22

14:                                               ; preds = %11
  %15 = load i32, ptr %7, align 4
  %16 = getelementptr inbounds [50 x i32], ptr %6, i32 0, i32 %15
  %17 = load i32, ptr %16, align 4
  %18 = call i32 (ptr, ...) @printf(ptr noundef @.str, i32 noundef %17)
  br label %19

19:                                               ; preds = %14
  %20 = load i32, ptr %7, align 4
  %21 = add nsw i32 %20, 1
  store i32 %21, ptr %7, align 4
  br label %11, !llvm.loop !11

22:                                               ; preds = %11
  %23 = call i32 (ptr, ...) @printf(ptr noundef @.str.1)
  %24 = call x86_stdcallcc i32 @"\01_QueryPerformanceCounter@4"(ptr noundef %4)
  %25 = load i64, ptr %4, align 8
  %26 = load i64, ptr %3, align 8
  %27 = sub nsw i64 %25, %26
  %28 = sitofp i64 %27 to double
  %29 = fmul double %28, 1.000000e+03
  %30 = load i64, ptr %2, align 8
  %31 = sitofp i64 %30 to double
  %32 = fdiv double %29, %31
  store double %32, ptr %5, align 8
  %33 = load double, ptr %5, align 8
  %34 = call i32 (ptr, ...) @printf(ptr noundef @.str.2, double noundef %33)
  ret i32 0
}

declare dllimport x86_stdcallcc i32 @"\01_QueryPerformanceFrequency@4"(ptr noundef) #1

declare dllimport x86_stdcallcc i32 @"\01_QueryPerformanceCounter@4"(ptr noundef) #1

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i32(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i32, i1 immarg) #2

; Function Attrs: noinline nounwind optnone
define internal i32 @printf(ptr noundef nonnull %0, ...) #0 {
  %2 = alloca ptr, align 4
  %3 = alloca i32, align 4
  %4 = alloca ptr, align 4
  store ptr %0, ptr %2, align 4
  call void @llvm.va_start(ptr %4)
  %5 = call ptr @__acrt_iob_func(i32 noundef 1)
  %6 = load ptr, ptr %2, align 4
  %7 = load ptr, ptr %4, align 4
  %8 = call i32 @__mingw_vfprintf(ptr noundef %5, ptr noundef %6, ptr noundef %7) #5
  store i32 %8, ptr %3, align 4
  call void @llvm.va_end(ptr %4)
  %9 = load i32, ptr %3, align 4
  ret i32 %9
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.va_start(ptr) #3

; Function Attrs: nounwind
declare dso_local i32 @__mingw_vfprintf(ptr noundef, ptr noundef, ptr noundef) #4

declare dllimport ptr @__acrt_iob_func(i32 noundef) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.va_end(ptr) #3

attributes #0 = { noinline nounwind optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="i686" "target-features"="+cmov,+cx8,+x87" "tune-cpu"="generic" }
attributes #1 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="i686" "target-features"="+cmov,+cx8,+x87" "tune-cpu"="generic" }
attributes #2 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { nocallback nofree nosync nounwind willreturn }
attributes #4 = { nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="i686" "target-features"="+cmov,+cx8,+x87" "tune-cpu"="generic" }
attributes #5 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"NumRegisterParameters", i32 0}
!1 = !{i32 1, !"wchar_size", i32 2}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 1, !"MaxTLSAlign", i32 65536}
!4 = !{!"Ubuntu clang version 18.1.3 (1ubuntu1)"}
!5 = distinct !{!5, !6}
!6 = !{!"llvm.loop.mustprogress"}
!7 = distinct !{!7, !6}
!8 = distinct !{!8, !6}
!9 = distinct !{!9, !6}
!10 = distinct !{!10, !6}
!11 = distinct !{!11, !6}
