/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#if defined(VISIONP6_AO)
#include "tensorflow/lite/micro/kernels/xtensa/conv.h"
#else
#include "tensorflow/lite/micro/kernels/conv.h"
#endif

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/conv.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/fixedpoint_utils.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_conv.h"

#if defined(VISIONP6_AO)
#include "vision_api.h"
#include "utils.h"
#endif

namespace tflite {
namespace {

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  
#if defined(VISIONP6_AO)  
  if (InitXtensaContext())
      return nullptr;
#endif
  
  return context->AllocatePersistentBuffer(context, sizeof(XtensaConvOpData));
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_OK(context, ConvPrepare(context, node));

#if defined(FUSION_F1) || defined(HIFI5)
  TF_LITE_ENSURE_OK(context, ConvPrepareHifi(context, node));
#endif
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

#if !defined(VISIONP6_AO)  
  const auto& params =
      *(reinterpret_cast<TfLiteConvParams*>(node->builtin_data));
#endif
  const auto& op_data = *(reinterpret_cast<XtensaConvOpData*>(node->user_data));

  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kConvOutputTensor);
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kConvInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 3)
          ? tflite::micro::GetEvalInput(context, node, kConvBiasTensor)
          : nullptr;

#if defined(HIFIMINI)
  int* input_dims = input->dims->data;
  int* filter_dims = filter->dims->data;
  if (input_dims[0] == 1 && input_dims[1] == 1 && input_dims[2] == 1 &&
      input_dims[3] == 32 && filter_dims[0] == 32 && filter_dims[1] == 1 &&
      filter_dims[2] == 1 && filter_dims[3] == 32) {
    Conv1x32Input32x32FilterHifiMini(
        -op_data.reference_op_data.input_zero_point,
        op_data.reference_op_data.output_zero_point,
        op_data.reference_op_data.output_activation_min,
        op_data.reference_op_data.output_activation_max,
        op_data.reference_op_data.per_channel_output_multiplier,
        op_data.reference_op_data.per_channel_output_shift,
        tflite::micro::GetTensorShape(input),
        tflite::micro::GetTensorData<int8_t>(input),
        tflite::micro::GetTensorShape(filter),
        tflite::micro::GetTensorData<int8_t>(filter),
        tflite::micro::GetTensorShape(bias),
        tflite::micro::GetTensorData<int32_t>(bias),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<int8_t>(output));
    return kTfLiteOk;
  }
#endif  // defined(HIFIMINI)

  switch (input->type) {
    case kTfLiteInt8: {
#if defined(HIFIMINI)
      ConvEvalHifiMini(ConvParamsQuantized(params, op_data.reference_op_data),
                       op_data.reference_op_data.per_channel_output_multiplier,
                       op_data.reference_op_data.per_channel_output_shift,
                       tflite::micro::GetTensorShape(input),
                       tflite::micro::GetTensorData<int8_t>(input),
                       tflite::micro::GetTensorShape(filter),
                       tflite::micro::GetTensorData<int8_t>(filter),
                       tflite::micro::GetTensorShape(bias),
                       tflite::micro::GetTensorData<int32_t>(bias),
                       tflite::micro::GetTensorShape(output),
                       tflite::micro::GetTensorData<int8_t>(output));
#elif defined(FUSION_F1) || defined(HIFI5)
      ConvEvalHifi(context, node, params, op_data, input, filter, bias, output);
#elif defined(VISIONP6_AO)
      if (op_data.reference_op_data.enableXtensaKernel) {
        uint32_t input_size = input->dims->data[0] * input->dims->data[1] *
                              input->dims->data[2] * input->dims->data[3];
        uint32_t output_size = output->dims->data[0] * output->dims->data[1] *
                              output->dims->data[2] * output->dims->data[3];
        uint32_t num_channels = filter->dims->data[kConvQuantizedDimension];

        xiConv(op_data.reference_op_data.pContext, op_data.reference_op_data.contextSize,
            (int8_t *)tflite::micro::GetTensorData<int8_t>(input), input_size,
            tflite::micro::GetTensorData<int8_t>(output), output_size,
            op_data.reference_op_data.reordCoeffnBias, op_data.reference_op_data.reordCoeffnBiasSize,
            op_data.reference_op_data.per_channel_output_multiplier,
            op_data.reference_op_data.per_channel_output_shift_int8, num_channels);
        break;
      }
#else      
      reference_integer_ops::ConvPerChannel(
          ConvParamsQuantized(params, op_data.reference_op_data),
          op_data.reference_op_data.per_channel_output_multiplier,
          op_data.reference_op_data.per_channel_output_shift,
          tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<int8_t>(input),
          tflite::micro::GetTensorShape(filter),
          tflite::micro::GetTensorData<int8_t>(filter),
          tflite::micro::GetTensorShape(bias),
          tflite::micro::GetTensorData<int32_t>(bias),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int8_t>(output));
#endif
      break;
    }
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                         TfLiteTypeGetName(input->type), input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}
}  // namespace

TfLiteRegistration Register_CONV_2D() {
  return {/*init=*/Init,
          /*free=*/nullptr,
          /*prepare=*/Prepare,
          /*invoke=*/Eval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace tflite
