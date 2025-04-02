; ModuleID = 'bin.c'
source_filename = "bin.c"
target datalayout = "e-m:x-p:32:32-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-w64-windows-gnu"

%struct.VideoHeader = type { i32, i32, i32 }
%union._LARGE_INTEGER = type { i64 }

@.str = private unnamed_addr constant [3 x i8] c"rb\00", align 1
@.str.1 = private unnamed_addr constant [3 x i8] c"wb\00", align 1
@.str.2 = private unnamed_addr constant [16 x i8] c"input_video.raw\00", align 1
@.str.3 = private unnamed_addr constant [17 x i8] c"output_video.raw\00", align 1
@.str.4 = private unnamed_addr constant [35 x i8] c"Time measured: %.3f milliseconds.\0A\00", align 1

; Function Attrs: noinline nounwind optnone
define dso_local void @rgb2ycbcr(ptr noundef %0, ptr noundef %1) #0 {
  %3 = alloca ptr, align 4
  %4 = alloca ptr, align 4
  %5 = alloca ptr, align 4
  %6 = alloca ptr, align 4
  %7 = alloca %struct.VideoHeader, align 4
  %8 = alloca i32, align 4
  %9 = alloca ptr, align 4
  %10 = alloca i32, align 4
  %11 = alloca ptr, align 4
  %12 = alloca i32, align 4
  %13 = alloca i8, align 1
  %14 = alloca i8, align 1
  %15 = alloca i8, align 1
  %16 = alloca i32, align 4
  %17 = alloca i32, align 4
  %18 = alloca i32, align 4
  %19 = alloca i32, align 4
  %20 = alloca i32, align 4
  %21 = alloca ptr, align 4
  %22 = alloca i32, align 4
  %23 = alloca i32, align 4
  %24 = alloca i32, align 4
  %25 = alloca i32, align 4
  %26 = alloca i32, align 4
  %27 = alloca i32, align 4
  %28 = alloca i32, align 4
  %29 = alloca i8, align 1
  %30 = alloca i8, align 1
  %31 = alloca i8, align 1
  %32 = alloca i32, align 4
  %33 = alloca i32, align 4
  %34 = alloca i32, align 4
  store ptr %0, ptr %3, align 4
  store ptr %1, ptr %4, align 4
  %35 = load ptr, ptr %3, align 4
  %36 = call ptr @fopen(ptr noundef %35, ptr noundef @.str)
  store ptr %36, ptr %5, align 4
  %37 = load ptr, ptr %5, align 4
  %38 = icmp ne ptr %37, null
  br i1 %38, label %40, label %39

39:                                               ; preds = %2
  br label %459

40:                                               ; preds = %2
  %41 = load ptr, ptr %4, align 4
  %42 = call ptr @fopen(ptr noundef %41, ptr noundef @.str.1)
  store ptr %42, ptr %6, align 4
  %43 = load ptr, ptr %6, align 4
  %44 = icmp ne ptr %43, null
  br i1 %44, label %48, label %45

45:                                               ; preds = %40
  %46 = load ptr, ptr %5, align 4
  %47 = call i32 @fclose(ptr noundef %46)
  br label %459

48:                                               ; preds = %40
  %49 = load ptr, ptr %5, align 4
  %50 = call i32 @fread(ptr noundef %7, i32 noundef 12, i32 noundef 1, ptr noundef %49)
  %51 = icmp ne i32 %50, 1
  br i1 %51, label %52, label %57

52:                                               ; preds = %48
  %53 = load ptr, ptr %5, align 4
  %54 = call i32 @fclose(ptr noundef %53)
  %55 = load ptr, ptr %6, align 4
  %56 = call i32 @fclose(ptr noundef %55)
  br label %459

57:                                               ; preds = %48
  %58 = getelementptr inbounds %struct.VideoHeader, ptr %7, i32 0, i32 0
  %59 = load i32, ptr %58, align 4
  %60 = getelementptr inbounds %struct.VideoHeader, ptr %7, i32 0, i32 1
  %61 = load i32, ptr %60, align 4
  %62 = mul i32 %59, %61
  %63 = mul i32 %62, 3
  store i32 %63, ptr %8, align 4
  %64 = load i32, ptr %8, align 4
  %65 = call ptr @malloc(i32 noundef %64) #6
  store ptr %65, ptr %9, align 4
  %66 = load ptr, ptr %9, align 4
  %67 = icmp ne ptr %66, null
  br i1 %67, label %73, label %68

68:                                               ; preds = %57
  %69 = load ptr, ptr %5, align 4
  %70 = call i32 @fclose(ptr noundef %69)
  %71 = load ptr, ptr %6, align 4
  %72 = call i32 @fclose(ptr noundef %71)
  br label %459

73:                                               ; preds = %57
  %74 = load ptr, ptr %6, align 4
  %75 = call i32 @fwrite(ptr noundef %7, i32 noundef 12, i32 noundef 1, ptr noundef %74)
  %76 = icmp ne i32 %75, 1
  br i1 %76, label %77, label %83

77:                                               ; preds = %73
  %78 = load ptr, ptr %9, align 4
  call void @free(ptr noundef %78)
  %79 = load ptr, ptr %5, align 4
  %80 = call i32 @fclose(ptr noundef %79)
  %81 = load ptr, ptr %6, align 4
  %82 = call i32 @fclose(ptr noundef %81)
  br label %459

83:                                               ; preds = %73
  store i32 0, ptr %10, align 4
  br label %84

84:                                               ; preds = %450, %83
  %85 = load i32, ptr %10, align 4
  %86 = getelementptr inbounds %struct.VideoHeader, ptr %7, i32 0, i32 2
  %87 = load i32, ptr %86, align 4
  %88 = icmp ult i32 %85, %87
  br i1 %88, label %89, label %453

89:                                               ; preds = %84
  %90 = load ptr, ptr %9, align 4
  %91 = load i32, ptr %8, align 4
  %92 = load ptr, ptr %5, align 4
  %93 = call i32 @fread(ptr noundef %90, i32 noundef 1, i32 noundef %91, ptr noundef %92)
  %94 = load i32, ptr %8, align 4
  %95 = icmp ne i32 %93, %94
  br i1 %95, label %96, label %97

96:                                               ; preds = %89
  br label %453

97:                                               ; preds = %89
  %98 = load i32, ptr %8, align 4
  %99 = call ptr @malloc(i32 noundef %98) #6
  store ptr %99, ptr %11, align 4
  %100 = load ptr, ptr %11, align 4
  %101 = icmp ne ptr %100, null
  br i1 %101, label %103, label %102

102:                                              ; preds = %97
  br label %453

103:                                              ; preds = %97
  store i32 0, ptr %12, align 4
  br label %104

104:                                              ; preds = %189, %103
  %105 = load i32, ptr %12, align 4
  %106 = getelementptr inbounds %struct.VideoHeader, ptr %7, i32 0, i32 0
  %107 = load i32, ptr %106, align 4
  %108 = getelementptr inbounds %struct.VideoHeader, ptr %7, i32 0, i32 1
  %109 = load i32, ptr %108, align 4
  %110 = mul i32 %107, %109
  %111 = icmp ult i32 %105, %110
  br i1 %111, label %112, label %192

112:                                              ; preds = %104
  %113 = load ptr, ptr %9, align 4
  %114 = load i32, ptr %12, align 4
  %115 = mul i32 %114, 3
  %116 = getelementptr inbounds i8, ptr %113, i32 %115
  %117 = load i8, ptr %116, align 1
  store i8 %117, ptr %13, align 1
  %118 = load ptr, ptr %9, align 4
  %119 = load i32, ptr %12, align 4
  %120 = mul i32 %119, 3
  %121 = add i32 %120, 1
  %122 = getelementptr inbounds i8, ptr %118, i32 %121
  %123 = load i8, ptr %122, align 1
  store i8 %123, ptr %14, align 1
  %124 = load ptr, ptr %9, align 4
  %125 = load i32, ptr %12, align 4
  %126 = mul i32 %125, 3
  %127 = add i32 %126, 2
  %128 = getelementptr inbounds i8, ptr %124, i32 %127
  %129 = load i8, ptr %128, align 1
  store i8 %129, ptr %15, align 1
  %130 = load i8, ptr %13, align 1
  %131 = zext i8 %130 to i32
  %132 = sitofp i32 %131 to double
  %133 = load i8, ptr %14, align 1
  %134 = zext i8 %133 to i32
  %135 = sitofp i32 %134 to double
  %136 = fmul double 5.870000e-01, %135
  %137 = call double @llvm.fmuladd.f64(double 2.990000e-01, double %132, double %136)
  %138 = load i8, ptr %15, align 1
  %139 = zext i8 %138 to i32
  %140 = sitofp i32 %139 to double
  %141 = call double @llvm.fmuladd.f64(double 1.140000e-01, double %140, double %137)
  %142 = fptosi double %141 to i32
  store i32 %142, ptr %16, align 4
  %143 = load i8, ptr %13, align 1
  %144 = zext i8 %143 to i32
  %145 = sitofp i32 %144 to double
  %146 = call double @llvm.fmuladd.f64(double -1.687360e-01, double %145, double 1.280000e+02)
  %147 = load i8, ptr %14, align 1
  %148 = zext i8 %147 to i32
  %149 = sitofp i32 %148 to double
  %150 = call double @llvm.fmuladd.f64(double -3.312640e-01, double %149, double %146)
  %151 = load i8, ptr %15, align 1
  %152 = zext i8 %151 to i32
  %153 = sitofp i32 %152 to double
  %154 = call double @llvm.fmuladd.f64(double 5.000000e-01, double %153, double %150)
  %155 = fptosi double %154 to i32
  store i32 %155, ptr %17, align 4
  %156 = load i8, ptr %13, align 1
  %157 = zext i8 %156 to i32
  %158 = sitofp i32 %157 to double
  %159 = call double @llvm.fmuladd.f64(double 5.000000e-01, double %158, double 1.280000e+02)
  %160 = load i8, ptr %14, align 1
  %161 = zext i8 %160 to i32
  %162 = sitofp i32 %161 to double
  %163 = call double @llvm.fmuladd.f64(double -4.186880e-01, double %162, double %159)
  %164 = load i8, ptr %15, align 1
  %165 = zext i8 %164 to i32
  %166 = sitofp i32 %165 to double
  %167 = call double @llvm.fmuladd.f64(double -8.131200e-02, double %166, double %163)
  %168 = fptosi double %167 to i32
  store i32 %168, ptr %18, align 4
  %169 = load i32, ptr %16, align 4
  %170 = trunc i32 %169 to i8
  %171 = load ptr, ptr %11, align 4
  %172 = load i32, ptr %12, align 4
  %173 = mul i32 %172, 3
  %174 = getelementptr inbounds i8, ptr %171, i32 %173
  store i8 %170, ptr %174, align 1
  %175 = load i32, ptr %17, align 4
  %176 = trunc i32 %175 to i8
  %177 = load ptr, ptr %11, align 4
  %178 = load i32, ptr %12, align 4
  %179 = mul i32 %178, 3
  %180 = add i32 %179, 1
  %181 = getelementptr inbounds i8, ptr %177, i32 %180
  store i8 %176, ptr %181, align 1
  %182 = load i32, ptr %18, align 4
  %183 = trunc i32 %182 to i8
  %184 = load ptr, ptr %11, align 4
  %185 = load i32, ptr %12, align 4
  %186 = mul i32 %185, 3
  %187 = add i32 %186, 2
  %188 = getelementptr inbounds i8, ptr %184, i32 %187
  store i8 %183, ptr %188, align 1
  br label %189

189:                                              ; preds = %112
  %190 = load i32, ptr %12, align 4
  %191 = add i32 %190, 1
  store i32 %191, ptr %12, align 4
  br label %104, !llvm.loop !5

192:                                              ; preds = %104
  %193 = getelementptr inbounds %struct.VideoHeader, ptr %7, i32 0, i32 0
  %194 = load i32, ptr %193, align 4
  %195 = udiv i32 %194, 2
  store i32 %195, ptr %19, align 4
  %196 = getelementptr inbounds %struct.VideoHeader, ptr %7, i32 0, i32 1
  %197 = load i32, ptr %196, align 4
  %198 = udiv i32 %197, 2
  store i32 %198, ptr %20, align 4
  %199 = load i32, ptr %19, align 4
  %200 = load i32, ptr %20, align 4
  %201 = mul i32 %199, %200
  %202 = mul i32 %201, 2
  %203 = call ptr @malloc(i32 noundef %202) #6
  store ptr %203, ptr %21, align 4
  %204 = load ptr, ptr %21, align 4
  %205 = icmp ne ptr %204, null
  br i1 %205, label %208, label %206

206:                                              ; preds = %192
  %207 = load ptr, ptr %11, align 4
  call void @free(ptr noundef %207)
  br label %453

208:                                              ; preds = %192
  store i32 0, ptr %22, align 4
  br label %209

209:                                              ; preds = %259, %208
  %210 = load i32, ptr %22, align 4
  %211 = load i32, ptr %20, align 4
  %212 = icmp ult i32 %210, %211
  br i1 %212, label %213, label %262

213:                                              ; preds = %209
  store i32 0, ptr %23, align 4
  br label %214

214:                                              ; preds = %255, %213
  %215 = load i32, ptr %23, align 4
  %216 = load i32, ptr %19, align 4
  %217 = icmp ult i32 %215, %216
  br i1 %217, label %218, label %258

218:                                              ; preds = %214
  %219 = load i32, ptr %22, align 4
  %220 = mul i32 %219, 2
  %221 = getelementptr inbounds %struct.VideoHeader, ptr %7, i32 0, i32 0
  %222 = load i32, ptr %221, align 4
  %223 = mul i32 %220, %222
  %224 = load i32, ptr %23, align 4
  %225 = mul i32 %224, 2
  %226 = add i32 %223, %225
  %227 = mul i32 %226, 3
  %228 = add i32 %227, 1
  store i32 %228, ptr %24, align 4
  %229 = load ptr, ptr %11, align 4
  %230 = load i32, ptr %24, align 4
  %231 = getelementptr inbounds i8, ptr %229, i32 %230
  %232 = load i8, ptr %231, align 1
  %233 = load ptr, ptr %21, align 4
  %234 = load i32, ptr %22, align 4
  %235 = load i32, ptr %19, align 4
  %236 = mul i32 %234, %235
  %237 = load i32, ptr %23, align 4
  %238 = add i32 %236, %237
  %239 = mul i32 %238, 2
  %240 = getelementptr inbounds i8, ptr %233, i32 %239
  store i8 %232, ptr %240, align 1
  %241 = load ptr, ptr %11, align 4
  %242 = load i32, ptr %24, align 4
  %243 = add i32 %242, 1
  %244 = getelementptr inbounds i8, ptr %241, i32 %243
  %245 = load i8, ptr %244, align 1
  %246 = load ptr, ptr %21, align 4
  %247 = load i32, ptr %22, align 4
  %248 = load i32, ptr %19, align 4
  %249 = mul i32 %247, %248
  %250 = load i32, ptr %23, align 4
  %251 = add i32 %249, %250
  %252 = mul i32 %251, 2
  %253 = add i32 %252, 1
  %254 = getelementptr inbounds i8, ptr %246, i32 %253
  store i8 %245, ptr %254, align 1
  br label %255

255:                                              ; preds = %218
  %256 = load i32, ptr %23, align 4
  %257 = add i32 %256, 1
  store i32 %257, ptr %23, align 4
  br label %214, !llvm.loop !7

258:                                              ; preds = %214
  br label %259

259:                                              ; preds = %258
  %260 = load i32, ptr %22, align 4
  %261 = add i32 %260, 1
  store i32 %261, ptr %22, align 4
  br label %209, !llvm.loop !8

262:                                              ; preds = %209
  store i32 0, ptr %25, align 4
  br label %263

263:                                              ; preds = %313, %262
  %264 = load i32, ptr %25, align 4
  %265 = load i32, ptr %20, align 4
  %266 = icmp ult i32 %264, %265
  br i1 %266, label %267, label %316

267:                                              ; preds = %263
  store i32 0, ptr %26, align 4
  br label %268

268:                                              ; preds = %309, %267
  %269 = load i32, ptr %26, align 4
  %270 = load i32, ptr %19, align 4
  %271 = icmp ult i32 %269, %270
  br i1 %271, label %272, label %312

272:                                              ; preds = %268
  %273 = load i32, ptr %25, align 4
  %274 = mul i32 %273, 2
  %275 = getelementptr inbounds %struct.VideoHeader, ptr %7, i32 0, i32 0
  %276 = load i32, ptr %275, align 4
  %277 = mul i32 %274, %276
  %278 = load i32, ptr %26, align 4
  %279 = mul i32 %278, 2
  %280 = add i32 %277, %279
  %281 = mul i32 %280, 3
  %282 = add i32 %281, 1
  store i32 %282, ptr %27, align 4
  %283 = load ptr, ptr %21, align 4
  %284 = load i32, ptr %25, align 4
  %285 = load i32, ptr %19, align 4
  %286 = mul i32 %284, %285
  %287 = load i32, ptr %26, align 4
  %288 = add i32 %286, %287
  %289 = mul i32 %288, 2
  %290 = getelementptr inbounds i8, ptr %283, i32 %289
  %291 = load i8, ptr %290, align 1
  %292 = load ptr, ptr %11, align 4
  %293 = load i32, ptr %27, align 4
  %294 = getelementptr inbounds i8, ptr %292, i32 %293
  store i8 %291, ptr %294, align 1
  %295 = load ptr, ptr %21, align 4
  %296 = load i32, ptr %25, align 4
  %297 = load i32, ptr %19, align 4
  %298 = mul i32 %296, %297
  %299 = load i32, ptr %26, align 4
  %300 = add i32 %298, %299
  %301 = mul i32 %300, 2
  %302 = add i32 %301, 1
  %303 = getelementptr inbounds i8, ptr %295, i32 %302
  %304 = load i8, ptr %303, align 1
  %305 = load ptr, ptr %11, align 4
  %306 = load i32, ptr %27, align 4
  %307 = add i32 %306, 1
  %308 = getelementptr inbounds i8, ptr %305, i32 %307
  store i8 %304, ptr %308, align 1
  br label %309

309:                                              ; preds = %272
  %310 = load i32, ptr %26, align 4
  %311 = add i32 %310, 1
  store i32 %311, ptr %26, align 4
  br label %268, !llvm.loop !9

312:                                              ; preds = %268
  br label %313

313:                                              ; preds = %312
  %314 = load i32, ptr %25, align 4
  %315 = add i32 %314, 1
  store i32 %315, ptr %25, align 4
  br label %263, !llvm.loop !10

316:                                              ; preds = %263
  %317 = load ptr, ptr %21, align 4
  call void @free(ptr noundef %317)
  store i32 0, ptr %28, align 4
  br label %318

318:                                              ; preds = %437, %316
  %319 = load i32, ptr %28, align 4
  %320 = getelementptr inbounds %struct.VideoHeader, ptr %7, i32 0, i32 0
  %321 = load i32, ptr %320, align 4
  %322 = getelementptr inbounds %struct.VideoHeader, ptr %7, i32 0, i32 1
  %323 = load i32, ptr %322, align 4
  %324 = mul i32 %321, %323
  %325 = icmp ult i32 %319, %324
  br i1 %325, label %326, label %440

326:                                              ; preds = %318
  %327 = load ptr, ptr %11, align 4
  %328 = load i32, ptr %28, align 4
  %329 = mul i32 %328, 3
  %330 = getelementptr inbounds i8, ptr %327, i32 %329
  %331 = load i8, ptr %330, align 1
  store i8 %331, ptr %29, align 1
  %332 = load ptr, ptr %11, align 4
  %333 = load i32, ptr %28, align 4
  %334 = mul i32 %333, 3
  %335 = add i32 %334, 1
  %336 = getelementptr inbounds i8, ptr %332, i32 %335
  %337 = load i8, ptr %336, align 1
  store i8 %337, ptr %30, align 1
  %338 = load ptr, ptr %11, align 4
  %339 = load i32, ptr %28, align 4
  %340 = mul i32 %339, 3
  %341 = add i32 %340, 2
  %342 = getelementptr inbounds i8, ptr %338, i32 %341
  %343 = load i8, ptr %342, align 1
  store i8 %343, ptr %31, align 1
  %344 = load i8, ptr %29, align 1
  %345 = zext i8 %344 to i32
  %346 = load i8, ptr %31, align 1
  %347 = zext i8 %346 to i32
  %348 = sub nsw i32 %347, 128
  %349 = sitofp i32 %348 to double
  %350 = fmul double 1.402000e+00, %349
  %351 = fptosi double %350 to i32
  %352 = add nsw i32 %345, %351
  store i32 %352, ptr %32, align 4
  %353 = load i8, ptr %29, align 1
  %354 = zext i8 %353 to i32
  %355 = load i8, ptr %30, align 1
  %356 = zext i8 %355 to i32
  %357 = sub nsw i32 %356, 128
  %358 = sitofp i32 %357 to double
  %359 = fmul double 3.441360e-01, %358
  %360 = fptosi double %359 to i32
  %361 = sub nsw i32 %354, %360
  %362 = load i8, ptr %31, align 1
  %363 = zext i8 %362 to i32
  %364 = sub nsw i32 %363, 128
  %365 = sitofp i32 %364 to double
  %366 = fmul double 7.141360e-01, %365
  %367 = fptosi double %366 to i32
  %368 = sub nsw i32 %361, %367
  store i32 %368, ptr %33, align 4
  %369 = load i8, ptr %29, align 1
  %370 = zext i8 %369 to i32
  %371 = load i8, ptr %30, align 1
  %372 = zext i8 %371 to i32
  %373 = sub nsw i32 %372, 128
  %374 = sitofp i32 %373 to double
  %375 = fmul double 1.772000e+00, %374
  %376 = fptosi double %375 to i32
  %377 = add nsw i32 %370, %376
  store i32 %377, ptr %34, align 4
  %378 = load i32, ptr %32, align 4
  %379 = icmp slt i32 %378, 0
  br i1 %379, label %380, label %381

380:                                              ; preds = %326
  br label %389

381:                                              ; preds = %326
  %382 = load i32, ptr %32, align 4
  %383 = icmp sgt i32 %382, 255
  br i1 %383, label %384, label %385

384:                                              ; preds = %381
  br label %387

385:                                              ; preds = %381
  %386 = load i32, ptr %32, align 4
  br label %387

387:                                              ; preds = %385, %384
  %388 = phi i32 [ 255, %384 ], [ %386, %385 ]
  br label %389

389:                                              ; preds = %387, %380
  %390 = phi i32 [ 0, %380 ], [ %388, %387 ]
  store i32 %390, ptr %32, align 4
  %391 = load i32, ptr %33, align 4
  %392 = icmp slt i32 %391, 0
  br i1 %392, label %393, label %394

393:                                              ; preds = %389
  br label %402

394:                                              ; preds = %389
  %395 = load i32, ptr %33, align 4
  %396 = icmp sgt i32 %395, 255
  br i1 %396, label %397, label %398

397:                                              ; preds = %394
  br label %400

398:                                              ; preds = %394
  %399 = load i32, ptr %33, align 4
  br label %400

400:                                              ; preds = %398, %397
  %401 = phi i32 [ 255, %397 ], [ %399, %398 ]
  br label %402

402:                                              ; preds = %400, %393
  %403 = phi i32 [ 0, %393 ], [ %401, %400 ]
  store i32 %403, ptr %33, align 4
  %404 = load i32, ptr %34, align 4
  %405 = icmp slt i32 %404, 0
  br i1 %405, label %406, label %407

406:                                              ; preds = %402
  br label %415

407:                                              ; preds = %402
  %408 = load i32, ptr %34, align 4
  %409 = icmp sgt i32 %408, 255
  br i1 %409, label %410, label %411

410:                                              ; preds = %407
  br label %413

411:                                              ; preds = %407
  %412 = load i32, ptr %34, align 4
  br label %413

413:                                              ; preds = %411, %410
  %414 = phi i32 [ 255, %410 ], [ %412, %411 ]
  br label %415

415:                                              ; preds = %413, %406
  %416 = phi i32 [ 0, %406 ], [ %414, %413 ]
  store i32 %416, ptr %34, align 4
  %417 = load i32, ptr %32, align 4
  %418 = trunc i32 %417 to i8
  %419 = load ptr, ptr %9, align 4
  %420 = load i32, ptr %28, align 4
  %421 = mul i32 %420, 3
  %422 = getelementptr inbounds i8, ptr %419, i32 %421
  store i8 %418, ptr %422, align 1
  %423 = load i32, ptr %33, align 4
  %424 = trunc i32 %423 to i8
  %425 = load ptr, ptr %9, align 4
  %426 = load i32, ptr %28, align 4
  %427 = mul i32 %426, 3
  %428 = add i32 %427, 1
  %429 = getelementptr inbounds i8, ptr %425, i32 %428
  store i8 %424, ptr %429, align 1
  %430 = load i32, ptr %34, align 4
  %431 = trunc i32 %430 to i8
  %432 = load ptr, ptr %9, align 4
  %433 = load i32, ptr %28, align 4
  %434 = mul i32 %433, 3
  %435 = add i32 %434, 2
  %436 = getelementptr inbounds i8, ptr %432, i32 %435
  store i8 %431, ptr %436, align 1
  br label %437

437:                                              ; preds = %415
  %438 = load i32, ptr %28, align 4
  %439 = add i32 %438, 1
  store i32 %439, ptr %28, align 4
  br label %318, !llvm.loop !11

440:                                              ; preds = %318
  %441 = load ptr, ptr %9, align 4
  %442 = load i32, ptr %8, align 4
  %443 = load ptr, ptr %6, align 4
  %444 = call i32 @fwrite(ptr noundef %441, i32 noundef 1, i32 noundef %442, ptr noundef %443)
  %445 = load i32, ptr %8, align 4
  %446 = icmp ne i32 %444, %445
  br i1 %446, label %447, label %448

447:                                              ; preds = %440
  br label %448

448:                                              ; preds = %447, %440
  %449 = load ptr, ptr %11, align 4
  call void @free(ptr noundef %449)
  br label %450

450:                                              ; preds = %448
  %451 = load i32, ptr %10, align 4
  %452 = add i32 %451, 1
  store i32 %452, ptr %10, align 4
  br label %84, !llvm.loop !12

453:                                              ; preds = %206, %102, %96, %84
  %454 = load ptr, ptr %9, align 4
  call void @free(ptr noundef %454)
  %455 = load ptr, ptr %5, align 4
  %456 = call i32 @fclose(ptr noundef %455)
  %457 = load ptr, ptr %6, align 4
  %458 = call i32 @fclose(ptr noundef %457)
  br label %459

459:                                              ; preds = %453, %77, %68, %52, %45, %39
  ret void
}

declare dso_local ptr @fopen(ptr noundef, ptr noundef) #1

declare dso_local i32 @fclose(ptr noundef) #1

declare dso_local i32 @fread(ptr noundef, i32 noundef, i32 noundef, ptr noundef) #1

; Function Attrs: allocsize(0)
declare dso_local ptr @malloc(i32 noundef) #2

declare dso_local i32 @fwrite(ptr noundef, i32 noundef, i32 noundef, ptr noundef) #1

declare dso_local void @free(ptr noundef) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fmuladd.f64(double, double, double) #3

; Function Attrs: noinline nounwind optnone
define dso_local i32 @main() #0 {
  %1 = alloca i32, align 4
  %2 = alloca %union._LARGE_INTEGER, align 8
  %3 = alloca %union._LARGE_INTEGER, align 8
  %4 = alloca %union._LARGE_INTEGER, align 8
  %5 = alloca double, align 8
  %6 = alloca ptr, align 4
  %7 = alloca ptr, align 4
  store i32 0, ptr %1, align 4
  %8 = call x86_stdcallcc i32 @"\01_QueryPerformanceFrequency@4"(ptr noundef %2)
  %9 = call x86_stdcallcc i32 @"\01_QueryPerformanceCounter@4"(ptr noundef %3)
  store ptr @.str.2, ptr %6, align 4
  store ptr @.str.3, ptr %7, align 4
  %10 = load ptr, ptr %6, align 4
  %11 = load ptr, ptr %7, align 4
  call void @rgb2ycbcr(ptr noundef %10, ptr noundef %11)
  %12 = call x86_stdcallcc i32 @"\01_QueryPerformanceCounter@4"(ptr noundef %4)
  %13 = load i64, ptr %4, align 8
  %14 = load i64, ptr %3, align 8
  %15 = sub nsw i64 %13, %14
  %16 = sitofp i64 %15 to double
  %17 = fmul double %16, 1.000000e+03
  %18 = load i64, ptr %2, align 8
  %19 = sitofp i64 %18 to double
  %20 = fdiv double %17, %19
  store double %20, ptr %5, align 8
  %21 = load double, ptr %5, align 8
  %22 = call i32 (ptr, ...) @printf(ptr noundef @.str.4, double noundef %21)
  ret i32 0
}

declare dllimport x86_stdcallcc i32 @"\01_QueryPerformanceFrequency@4"(ptr noundef) #1

declare dllimport x86_stdcallcc i32 @"\01_QueryPerformanceCounter@4"(ptr noundef) #1

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
  %8 = call i32 @__mingw_vfprintf(ptr noundef %5, ptr noundef %6, ptr noundef %7) #7
  store i32 %8, ptr %3, align 4
  call void @llvm.va_end(ptr %4)
  %9 = load i32, ptr %3, align 4
  ret i32 %9
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.va_start(ptr) #4

; Function Attrs: nounwind
declare dso_local i32 @__mingw_vfprintf(ptr noundef, ptr noundef, ptr noundef) #5

declare dllimport ptr @__acrt_iob_func(i32 noundef) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.va_end(ptr) #4

attributes #0 = { noinline nounwind optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="i686" "target-features"="+cmov,+cx8,+x87" "tune-cpu"="generic" }
attributes #1 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="i686" "target-features"="+cmov,+cx8,+x87" "tune-cpu"="generic" }
attributes #2 = { allocsize(0) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="i686" "target-features"="+cmov,+cx8,+x87" "tune-cpu"="generic" }
attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { nocallback nofree nosync nounwind willreturn }
attributes #5 = { nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="i686" "target-features"="+cmov,+cx8,+x87" "tune-cpu"="generic" }
attributes #6 = { allocsize(0) }
attributes #7 = { nounwind }

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
!12 = distinct !{!12, !6}
